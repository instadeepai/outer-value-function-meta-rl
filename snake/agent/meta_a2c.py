import functools
from typing import Dict, Optional, Protocol, Tuple

import chex
import jax
import jumanji
import optax
import scipy
from jax import numpy as jnp

from snake.agent.a2c import a2c_inner_loss, a2c_outer_loss
from snake.agent.actor_critic_agent import ActorCriticAgent
from snake.networks.actor_critic import ActorCriticNetworks
from snake.training.types import (
    ActingState,
    ActorCriticParams,
    HyperParams,
    Metal,
    MetaParams,
    State,
    TrainingState,
)


class MetaLoss(Protocol):
    def __call__(
        self, meta_params: MetaParams, state: State
    ) -> Tuple[jnp.float_, Tuple[TrainingState, ActingState, Dict]]:
        pass


class MetaA2C(ActorCriticAgent):
    def __init__(
        self,
        n_steps: int,
        total_batch_size: int,
        total_num_envs: int,
        env: jumanji.Environment,
        actor_critic_networks: ActorCriticNetworks,
        optimizer: optax.GradientTransformation,
        bootstrap_l_optimizer: Optional[optax.GradientTransformation],
        meta_optimizer: optax.GradientTransformation,
        normalize_advantage: bool,
        normalize_outer_advantage: Optional[bool],
        reward_scaling: float,
        hyper_params_init: HyperParams,
        outer_hyper_params: HyperParams,
        meta_objective: Optional[str],
        bootstrap_l: Optional[int],
        bootstrap_pm: Optional[float],
        bootstrap_vm: Optional[float],
        metal: Metal,
        l_kl_outer: Optional[float],
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
        self._meta_optimizer = meta_optimizer
        assert normalize_outer_advantage is not None
        self._normalize_outer_advantage = normalize_outer_advantage
        self._hyper_params_init = hyper_params_init
        self._outer_hyper_params = outer_hyper_params
        assert meta_objective in ["meta_gradient", "bootstrap"]
        self._meta_loss = make_meta_loss(self, meta_objective)
        self._metal = metal
        if meta_objective == "bootstrap":
            assert bootstrap_l is not None
            assert bootstrap_pm is not None
            assert bootstrap_vm is not None
        self.meta_objective = meta_objective
        self._bootstrap_l = bootstrap_l
        self._bootstrap_pm = bootstrap_pm
        self._bootstrap_vm = bootstrap_vm
        if meta_objective == "meta_gradient":
            assert l_kl_outer is not None
        self._l_kl_outer = l_kl_outer
        self._bootstrap_l_optimizer = bootstrap_l_optimizer

    @property
    def meta_optimizer(self) -> optax.GradientTransformation:
        return self._meta_optimizer

    @property
    def normalize_outer_advantage(self) -> bool:
        return self._normalize_outer_advantage

    @property
    def hyper_params_init(self) -> MetaParams:
        return self._hyper_params_init

    @property
    def outer_hyper_params(self) -> HyperParams:
        return self._outer_hyper_params

    @property
    def meta_loss(self) -> MetaLoss:
        return self._meta_loss

    @property
    def bootstrap_l(self) -> int:
        return self._bootstrap_l  # type: ignore

    @property
    def bootstrap_pm(self) -> Optional[float]:
        return self._bootstrap_pm

    @property
    def bootstrap_vm(self) -> Optional[float]:
        return self._bootstrap_vm

    @property
    def metal(self) -> Metal:
        return self._metal

    @property
    def l_kl_outer(self) -> Optional[float]:
        return self._l_kl_outer

    @property
    def bootstrap_l_optimizer(self) -> optax.GradientTransformation:
        return self._bootstrap_l_optimizer

    def init(self, key: chex.PRNGKey) -> State:
        params, acting_state = self.actor_critic_init(key)
        optimizer_state = self.optimizer.init(params)
        if (
            self.bootstrap_l_optimizer is not None
            and self.meta_objective == "bootstrap"
        ):
            bootstrap_l_optimizer_state = self.bootstrap_l_optimizer.init(params)
            optimizer_state = (optimizer_state, bootstrap_l_optimizer_state)
        meta_params = initialize_meta_params(self.hyper_params_init)
        meta_optimizer_state = self.meta_optimizer.init(meta_params)
        training_state = TrainingState(
            params=params,
            meta_params=meta_params,
            optimizer_state=optimizer_state,
            meta_optimizer_state=meta_optimizer_state,
            env_steps=jnp.int32(0),
        )
        training_state = jax.device_put_replicated(training_state, jax.local_devices())
        return State(training_state=training_state, acting_state=acting_state)

    def update(self, state: State) -> Tuple[State, Dict]:
        meta_grad, (new_training_state, new_acting_state, metrics) = jax.grad(
            self.meta_loss, argnums=0, has_aux=True
        )(
            state.training_state.meta_params,
            state,
        )
        meta_grad, metrics = jax.tree_util.tree_map(
            functools.partial(jax.lax.pmean, axis_name="devices"), (meta_grad, metrics)
        )
        meta_updates, meta_optimizer_state = self.meta_optimizer.update(
            meta_grad, state.training_state.meta_optimizer_state
        )
        meta_params: MetaParams = optax.apply_updates(
            state.training_state.meta_params, meta_updates
        )
        next_training_state = TrainingState(
            params=new_training_state.params,
            meta_params=meta_params,
            optimizer_state=new_training_state.optimizer_state,
            meta_optimizer_state=meta_optimizer_state,
            env_steps=new_training_state.env_steps,
        )
        next_state = State(
            training_state=next_training_state,
            acting_state=new_acting_state,
        )
        hyper_params = meta_params_to_hyper_params(meta_params)
        metrics.update(
            meta_grad_norm=optax.global_norm(meta_grad),
            meta_grad_gamma=meta_grad.gamma,
            meta_grad_lambda_=meta_grad.lambda_,
            **hyper_params._asdict(),
        )
        return next_state, metrics


def initialize_meta_params(hyper_params_init: HyperParams) -> MetaParams:
    """Initialize meta-params to logit(hyper-params)."""
    logits_hyper_params: HyperParams = jax.tree_util.tree_map(
        scipy.special.logit, hyper_params_init
    )
    meta_params = MetaParams(**logits_hyper_params._asdict())
    return meta_params


def meta_params_to_hyper_params(meta_params: MetaParams) -> HyperParams:
    """Apply sigmoid function to meta-params to get the hyper-params."""
    sigmoid_meta_params: MetaParams = jax.tree_util.tree_map(
        jax.nn.sigmoid, meta_params
    )
    hyper_params = HyperParams(**sigmoid_meta_params._asdict())
    return hyper_params


Carry = Tuple[ActorCriticParams, optax.OptState, ActingState]


def make_meta_loss(agent: MetaA2C, meta_objective: str) -> MetaLoss:  # noqa: CCR001
    def parameter_update(
        carry: Carry,
        hyper_params: HyperParams,
        normalizing_advantage: bool,
        optimizer: optax.GradientTransformation,
        outer_loss: bool = False,
    ) -> Tuple[Carry, Dict]:
        params, optimizer_state, acting_state = carry
        new_acting_state, data = agent.rollout(
            actor_params=params.actor,
            discount_factor=hyper_params.gamma,
            acting_state=acting_state,
        )
        chex.assert_tree_shape_prefix(
            data, (agent.n_steps, agent.batch_size_per_device)
        )
        entropy_key = new_acting_state.acting_key
        if not outer_loss:
            grad, metrics = jax.grad(a2c_inner_loss, argnums=0, has_aux=True)(
                params,
                hyper_params,
                agent.actor_critic_networks,
                data,
                entropy_key,
                normalizing_advantage,
                agent.reward_scaling,
                agent.outer_hyper_params,
            )
            grad, metrics = jax.tree_util.tree_map(
                functools.partial(jax.lax.pmean, axis_name="devices"), (grad, metrics)
            )
            metrics.update(grad_norm=optax.global_norm(grad))
        else:
            grad, metrics = jax.grad(a2c_outer_loss, argnums=0, has_aux=True)(
                params,
                agent.outer_hyper_params,
                agent.actor_critic_networks,
                data,
                entropy_key,
                normalizing_advantage,
                agent.reward_scaling,
            )
            grad, metrics = jax.tree_util.tree_map(
                functools.partial(jax.lax.pmean, axis_name="devices"), (grad, metrics)
            )
            metrics.update(grad_norm_outer_loss=optax.global_norm(grad))
        updates, new_optimizer_state = optimizer.update(grad, optimizer_state)
        new_params: ActorCriticParams = optax.apply_updates(params, updates)
        carry = (
            new_params,
            new_optimizer_state,
            new_acting_state,
        )
        return carry, metrics

    def meta_gradient_loss(
        meta_params: MetaParams, state: State
    ) -> Tuple[jnp.float_, Tuple[TrainingState, ActingState, Dict]]:
        # stop-gradient of meta-parameters that one does not want to learn
        meta_params = apply_metal_filter(meta_params, agent.metal)
        hyper_params = meta_params_to_hyper_params(meta_params)

        # inner gradient step
        carry = (
            state.training_state.params,
            state.training_state.optimizer_state,
            state.acting_state,
        )
        (
            (
                new_params,
                new_optimizer_state,
                new_acting_state,
            ),
            metrics,
        ) = parameter_update(
            carry, hyper_params, agent.normalize_advantage, agent.optimizer
        )
        new_env_steps = state.training_state.env_steps + jax.lax.psum(
            agent.n_steps * agent.batch_size_per_device, axis_name="devices"
        )
        metrics.update(env_steps=new_env_steps)

        # outer batch
        new_acting_state, data_outer = agent.rollout(
            actor_params=new_params.actor,
            discount_factor=agent.outer_hyper_params.gamma,  # Warning: the policy has not been trained to condition on gamma'
            acting_state=new_acting_state,
        )
        chex.assert_tree_shape_prefix(
            data_outer, (agent.n_steps, agent.batch_size_per_device)
        )
        new_training_state = TrainingState(
            optimizer_state=new_optimizer_state,
            meta_optimizer_state=None,
            params=new_params,
            meta_params=None,
            env_steps=new_env_steps,
        )
        entropy_key = new_acting_state.acting_key
        meta_loss, metrics_outer = a2c_outer_loss(
            new_params,
            agent.outer_hyper_params,
            agent.actor_critic_networks,
            data_outer,
            entropy_key,
            agent.normalize_outer_advantage,
            agent.reward_scaling,
        )
        # Add KL divergence term from STAC
        parametric_action_distribution = (
            agent.actor_critic_networks.parametric_action_distribution
        )
        policy_apply = agent.actor_critic_networks.policy_network.apply
        new_policy_logits = policy_apply(
            new_params.actor,
            data_outer.observation,
            agent.outer_hyper_params.gamma,
        )
        policy_logits = policy_apply(
            state.training_state.params.actor,
            data_outer.observation,
            agent.outer_hyper_params.gamma,  # Not sure what gamma to give here.
        )
        kl_divergence_loss = jnp.mean(
            parametric_action_distribution.kl_divergence(
                new_policy_logits, policy_logits
            )
        )
        meta_loss += agent.l_kl_outer * kl_divergence_loss
        metrics_outer.update(
            total_loss_outer_loss=meta_loss,
            kl_divergence_loss_outer_loss=kl_divergence_loss,
        )
        metrics.update(metrics_outer)
        return meta_loss, (new_training_state, new_acting_state, metrics)

    def bootstrap_loss(
        meta_params: MetaParams, state: State
    ) -> Tuple[jnp.float_, Tuple[TrainingState, ActingState, Dict]]:
        # stop-gradient of meta-parameters that one does not want to learn
        meta_params = apply_metal_filter(meta_params, agent.metal)
        hyper_params = meta_params_to_hyper_params(meta_params)

        if agent.bootstrap_l_optimizer is not None:
            optimizer_state, _ = state.training_state.optimizer_state
        else:
            optimizer_state = state.training_state.optimizer_state
        # inner gradient step
        carry = (
            state.training_state.params,
            optimizer_state,
            state.acting_state,
        )
        carry, metrics = (
            (
                new_params,
                new_optimizer_state,
                new_acting_state,
            ),
            metrics,
        ) = parameter_update(
            carry, hyper_params, agent.normalize_advantage, agent.optimizer
        )
        new_env_steps = state.training_state.env_steps + jax.lax.psum(
            agent.n_steps * agent.batch_size_per_device, axis_name="devices"
        )
        metrics.update(env_steps=new_env_steps)

        # L - 1 steps
        (target_params, target_optimizer_state, target_acting_state,), _ = jax.lax.scan(
            lambda c, _: parameter_update(
                c, hyper_params, agent.normalize_advantage, agent.optimizer
            ),
            carry,
            None,
            length=agent.bootstrap_l - 1,
        )

        # Lth step
        if agent.bootstrap_l_optimizer is not None:
            _, bootstrap_l_optimizer_state = state.training_state.optimizer_state
            bootstrap_l_optimizer = agent.bootstrap_l_optimizer
        else:
            bootstrap_l_optimizer_state = target_optimizer_state
            bootstrap_l_optimizer = agent.optimizer
        (
            (
                target_params,
                new_bootstrap_l_optimizer_state,
                target_acting_state,
            ),
            l_step_metrics,
        ) = parameter_update(
            (target_params, bootstrap_l_optimizer_state, target_acting_state),
            agent.outer_hyper_params,
            agent.normalize_outer_advantage,
            bootstrap_l_optimizer,
            outer_loss=True,
        )
        metrics.update(l_step_metrics)

        # Compute matching loss
        new_acting_state, data_outer = agent.rollout(
            actor_params=target_params.actor,
            discount_factor=agent.outer_hyper_params.gamma,  # Warning: the policy has not been trained to condition on this gamma'
            acting_state=target_acting_state,
        )
        chex.assert_tree_shape_prefix(
            data_outer, (agent.n_steps, agent.batch_size_per_device)
        )
        if agent.bootstrap_l_optimizer is not None:
            new_optimizer_state = (new_optimizer_state, new_bootstrap_l_optimizer_state)
        new_training_state = TrainingState(
            optimizer_state=new_optimizer_state,
            meta_optimizer_state=None,
            params=new_params,
            meta_params=None,
            env_steps=new_env_steps,
        )

        parametric_action_distribution = (
            agent.actor_critic_networks.parametric_action_distribution
        )
        policy_apply = agent.actor_critic_networks.policy_network.apply
        value_apply = agent.actor_critic_networks.value_network.apply
        new_policy_logits = policy_apply(
            new_params.actor,
            data_outer.observation,
            agent.outer_hyper_params.gamma,
        )
        target_policy_logits = policy_apply(
            jax.lax.stop_gradient(target_params.actor),
            data_outer.observation,
            agent.outer_hyper_params.gamma,
        )
        new_value = value_apply(
            new_params.critic,
            data_outer.observation,
            agent.outer_hyper_params.gamma,
        )
        target_value = value_apply(
            jax.lax.stop_gradient(target_params.critic),
            data_outer.observation,
            agent.outer_hyper_params.gamma,
        )

        kl_divergence = parametric_action_distribution.kl_divergence(
            target_policy_logits, new_policy_logits
        )
        policy_matching = jnp.mean(kl_divergence)
        value_matching = jnp.mean((target_value - new_value) ** 2)
        meta_loss = (
            agent.bootstrap_pm * policy_matching + agent.bootstrap_vm * value_matching
        )
        metrics.update(
            meta_loss=meta_loss,
            policy_matching=policy_matching,
            value_matching=value_matching,
        )
        return meta_loss, (new_training_state, new_acting_state, metrics)

    return meta_gradient_loss if meta_objective == "meta_gradient" else bootstrap_loss


def apply_metal_filter(meta_params: MetaParams, metal: Metal) -> MetaParams:
    meta_params = MetaParams(
        **jax.tree_util.tree_map(
            lambda f, m: f(m), metal._asdict(), meta_params._asdict()
        )
    )
    return meta_params
