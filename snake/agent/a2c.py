import functools
from typing import Dict, Optional, Tuple

import chex
import jax
import jumanji
import optax
from jax import numpy as jnp

from snake.agent.actor_critic_agent import ActorCriticAgent
from snake.agent.gae import compute_td_lambda
from snake.networks.actor_critic import ActorCriticNetworks
from snake.training.types import (
    ActorCriticParams,
    HyperParams,
    State,
    TrainingState,
    Transition,
)


class A2C(ActorCriticAgent):
    def __init__(
        self,
        n_steps: int,
        total_batch_size: int,
        total_num_envs: int,
        env: jumanji.Environment,
        actor_critic_networks: ActorCriticNetworks,
        optimizer: optax.GradientTransformation,
        normalize_advantage: bool,
        reward_scaling: float,
        hyper_params: HyperParams,
        env_type: str,
    ):
        super().__init__(
            n_steps,
            total_batch_size,
            total_num_envs,
            env,
            actor_critic_networks,
            optimizer,
            normalize_advantage,
            reward_scaling,
            env_type,
        )
        self._hyper_params = hyper_params

    @property
    def hyper_params(self) -> HyperParams:
        return self._hyper_params

    def init(self, key: chex.PRNGKey) -> State:
        params, acting_state = self.actor_critic_init(key)
        optimizer_state = self.optimizer.init(params)
        training_state = TrainingState(
            params=params,
            meta_params=None,
            optimizer_state=optimizer_state,
            meta_optimizer_state=None,
            env_steps=jnp.int32(0),
        )
        training_state = jax.device_put_replicated(training_state, jax.local_devices())
        return State(training_state=training_state, acting_state=acting_state)

    def update(self, state: State) -> Tuple[State, Dict]:
        training_state = state.training_state
        if training_state.meta_params is not None:
            discount_factor = jax.nn.sigmoid(training_state.meta_params.gamma)
        else:
            discount_factor = None
        next_acting_state, data = self.rollout(
            actor_params=training_state.params.actor,
            discount_factor=discount_factor,
            acting_state=state.acting_state,
        )
        chex.assert_tree_shape_prefix(data, (self.n_steps, self.batch_size_per_device))
        entropy_key = next_acting_state.acting_key
        grad, metrics = jax.grad(a2c_inner_loss, argnums=0, has_aux=True)(
            training_state.params,
            self.hyper_params,
            self.actor_critic_networks,
            data,
            entropy_key,
            self.normalize_advantage,
            self.reward_scaling,
            self.hyper_params,
        )
        grad, metrics = jax.tree_util.tree_map(
            functools.partial(jax.lax.pmean, axis_name="devices"), (grad, metrics)
        )
        updates, optimizer_state = self.optimizer.update(
            grad, training_state.optimizer_state
        )
        params: ActorCriticParams = optax.apply_updates(training_state.params, updates)
        new_env_steps = training_state.env_steps + jax.lax.psum(
            self.n_steps * self.batch_size_per_device, axis_name="devices"
        )
        next_training_state = TrainingState(
            optimizer_state=optimizer_state,
            meta_optimizer_state=None,
            params=params,
            meta_params=None,
            env_steps=new_env_steps,
        )
        next_state = State(
            training_state=next_training_state,
            acting_state=next_acting_state,
        )
        metrics.update(
            grad_norm=optax.global_norm(grad),
            env_steps=new_env_steps,
            **self.hyper_params._asdict(),
        )
        return next_state, metrics


def a2c_inner_loss(
    params: ActorCriticParams,
    hyper_params: HyperParams,
    actor_critic_networks: ActorCriticNetworks,
    data: Transition,
    entropy_key: chex.PRNGKey,
    normalize_advantage: bool,
    reward_scaling: float,
    outer_hyper_params: Optional[HyperParams],
) -> Tuple[jnp.float_, Dict]:
    """Assumes time dimension is first."""

    parametric_action_distribution = (
        actor_critic_networks.parametric_action_distribution
    )
    policy_apply = actor_critic_networks.policy_network.apply
    value_apply = actor_critic_networks.value_network.apply

    policy_logits = policy_apply(
        params.actor,
        data.observation,
        jax.lax.stop_gradient(hyper_params.gamma),
    )
    baseline = value_apply(
        params.critic,
        data.observation,
        jax.lax.stop_gradient(hyper_params.gamma),
    )
    bootstrap_value = value_apply(
        params.critic,
        get_last_observation(data.next_observation),
        jax.lax.stop_gradient(hyper_params.gamma),
    )

    rewards = reward_scaling * data.reward

    vs, advantages = compute_td_lambda(
        discount=data.discount,
        rewards=rewards,
        values=jax.lax.stop_gradient(baseline),
        bootstrap_value=jax.lax.stop_gradient(bootstrap_value),
        lambda_=hyper_params.lambda_,
        discount_factor=hyper_params.gamma,
    )

    metrics: Dict = {}
    if normalize_advantage:
        metrics.update(unnormalized_advantage=jnp.mean(advantages))
        advantages = safe_meta_gradient_standardise(advantages)

    metrics.update(
        advantage=jnp.mean(advantages),
        value=jnp.mean(baseline),
    )

    outer_critic = actor_critic_networks.outer_value_network is not None
    if outer_critic:
        assert actor_critic_networks.outer_value_network is not None
        assert outer_hyper_params is not None
        outer_critic_value_apply = actor_critic_networks.outer_value_network.apply
        outer_critic_baseline = outer_critic_value_apply(
            params.outer_critic,
            data.observation,
            outer_hyper_params.gamma,
        )
        outer_critic_bootstrap_value = outer_critic_value_apply(
            params.outer_critic,
            get_last_observation(data.next_observation),
            outer_hyper_params.gamma,
        )
        outer_critic_vs, _ = compute_td_lambda(
            discount=data.discount,
            rewards=rewards,
            values=jax.lax.stop_gradient(outer_critic_baseline),
            bootstrap_value=jax.lax.stop_gradient(outer_critic_bootstrap_value),
            lambda_=outer_hyper_params.lambda_,
            discount_factor=outer_hyper_params.gamma,
        )

        metrics.update(
            outer_critic_value=jnp.mean(outer_critic_baseline),
        )
        outer_critic_loss = jnp.mean((outer_critic_vs - outer_critic_baseline) ** 2)
    else:
        outer_critic_loss = jnp.array(0, jnp.float32)

    raw_actions = data.extras["policy_extras"]["raw_action"]
    log_probs = parametric_action_distribution.log_prob(policy_logits, raw_actions)
    policy_loss = -jnp.mean(advantages * log_probs)

    critic_loss = jnp.mean((vs - baseline) ** 2)

    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, entropy_key)
    )
    entropy_loss = -entropy

    total_loss = (
        hyper_params.l_pg * policy_loss
        + hyper_params.l_td * critic_loss
        + jax.lax.stop_gradient(hyper_params.l_td) * outer_critic_loss
        + hyper_params.l_en * entropy_loss
    )
    metrics.update(
        total_loss=total_loss,
        policy_loss=policy_loss,
        critic_loss=critic_loss,
        outer_critic_loss=outer_critic_loss,
        entropy_loss=entropy_loss,
        entropy=entropy,
    )
    return total_loss, metrics


def a2c_outer_loss(
    params: ActorCriticParams,
    outer_hyper_params: HyperParams,
    actor_critic_networks: ActorCriticNetworks,
    data: Transition,
    entropy_key: chex.PRNGKey,
    normalize_advantage: bool,
    reward_scaling: float,
) -> Tuple[jnp.float_, Dict]:
    """Assumes time dimension is first."""

    parametric_action_distribution = (
        actor_critic_networks.parametric_action_distribution
    )
    policy_apply = actor_critic_networks.policy_network.apply
    value_apply = actor_critic_networks.value_network.apply

    policy_logits = policy_apply(
        params.actor,
        data.observation,
        outer_hyper_params.gamma,  # Warning: the policy has not been trained to condition on gamma'
    )
    baseline = value_apply(
        params.critic,
        data.observation,
        outer_hyper_params.gamma,
    )
    bootstrap_value = value_apply(
        params.critic,
        get_last_observation(data.next_observation),
        outer_hyper_params.gamma,
    )

    rewards = reward_scaling * data.reward

    vs, advantages = compute_td_lambda(
        discount=data.discount,
        rewards=rewards,
        values=jax.lax.stop_gradient(baseline),
        bootstrap_value=jax.lax.stop_gradient(bootstrap_value),
        lambda_=outer_hyper_params.lambda_,
        discount_factor=outer_hyper_params.gamma,
    )

    metrics: Dict = {}
    metrics.update(
        value=jnp.mean(baseline),
    )

    outer_critic = actor_critic_networks.outer_value_network is not None
    if outer_critic:
        assert actor_critic_networks.outer_value_network is not None
        assert outer_hyper_params is not None
        outer_critic_value_apply = actor_critic_networks.outer_value_network.apply
        outer_critic_baseline = outer_critic_value_apply(
            params.outer_critic,
            data.observation,
            outer_hyper_params.gamma,
        )
        outer_critic_bootstrap_value = outer_critic_value_apply(
            params.outer_critic,
            get_last_observation(data.next_observation),
            outer_hyper_params.gamma,
        )
        _, outer_critic_advantages = compute_td_lambda(
            discount=data.discount,
            rewards=rewards,
            values=jax.lax.stop_gradient(outer_critic_baseline),
            bootstrap_value=jax.lax.stop_gradient(outer_critic_bootstrap_value),
            lambda_=outer_hyper_params.lambda_,
            discount_factor=outer_hyper_params.gamma,
        )

        metrics.update(
            outer_critic_value=jnp.mean(outer_critic_baseline),
        )
        advantages = outer_critic_advantages

    if normalize_advantage:
        metrics.update(unnormalized_advantage=jnp.mean(advantages))
        advantages = safe_meta_gradient_standardise(advantages)

    metrics.update(
        advantage=jnp.mean(advantages),
    )

    raw_actions = data.extras["policy_extras"]["raw_action"]
    log_probs = parametric_action_distribution.log_prob(policy_logits, raw_actions)
    policy_loss = -jnp.mean(advantages * log_probs)

    critic_loss = jnp.mean((vs - baseline) ** 2)

    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, entropy_key)
    )
    entropy_loss = -entropy

    total_loss = (
        outer_hyper_params.l_pg * policy_loss
        + outer_hyper_params.l_td * critic_loss
        + outer_hyper_params.l_en * entropy_loss
    )
    metrics.update(
        total_loss=total_loss,
        policy_loss=policy_loss,
        critic_loss=critic_loss,
        entropy_loss=entropy_loss,
        entropy=entropy,
    )
    metrics = {key + "_outer_loss": value for key, value in metrics.items()}
    return total_loss, metrics


def get_last_observation(observation: chex.ArrayTree) -> chex.ArrayTree:
    return jax.tree_util.tree_map(lambda x: x[-1], observation)


def safe_meta_gradient_standardise(advantages: chex.Array) -> chex.Array:
    # Normalize advantages across their 2 dimensions (time, batch)
    assert advantages.ndim == 2
    epsilon = jnp.array(1e-5, jnp.float32)
    mean = jnp.mean(advantages, axis=None, keepdims=True)
    variance = jnp.mean(jnp.square(advantages), axis=None, keepdims=True) - jnp.square(
        mean
    )
    normalized_advantages = (
        advantages - jax.lax.stop_gradient(mean)
    ) * jax.lax.stop_gradient(jax.lax.rsqrt(variance + epsilon))
    return normalized_advantages
