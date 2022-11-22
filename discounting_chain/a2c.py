from typing import Callable, NamedTuple, Optional, Tuple

import chex
import jax.random
import optax
import rlax
from jax import numpy as jnp

from discounting_chain.base import DataGeneratorState, Metrics, OnlineAgent
from discounting_chain.data_generation.n_step_data_generator import (
    make_n_step_data_generator,
)


class ExtrasA2C(NamedTuple):
    log_prob: chex.Array
    value: chex.Array
    entropy: chex.Array
    probs: Optional[chex.Array] = ()


class AgentState(NamedTuple):
    params: chex.ArrayTree
    opt_state: optax.OptState


class State(NamedTuple):
    agent_state: AgentState
    data_generator_state: DataGeneratorState


def safe_meta_gradient_standardise(advantages: chex.Array) -> chex.Array:
    # Normalize advantages across their 2 dimensions (time, batch)
    epsilon = jnp.array(1e-5, jnp.float32)
    mean = jnp.mean(advantages, axis=None, keepdims=True)
    variance = jnp.mean(jnp.square(advantages), axis=None, keepdims=True) - jnp.square(
        mean
    )
    normalized_advantages = (
        advantages - jax.lax.stop_gradient(mean)
    ) * jax.lax.stop_gradient(jax.lax.rsqrt(variance + epsilon))
    return normalized_advantages


def a2c_loss_and_metrics(
    log_prob_tm1,
    entropy,
    v_tm1,
    r_t,
    discount_t,
    v_t,
    discount: float,
    lambda_gae: float,
    entropy_cost: float,
    policy_grad_cost: float,
    critic_cost: float,
    normalise: bool = True,
) -> Tuple[chex.Scalar, Metrics]:
    discount_t = discount_t * discount
    target_tm1 = jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None))(
        r_t, discount_t, jax.lax.stop_gradient(v_t), lambda_gae
    )
    adv_t_policy = target_tm1 - jax.lax.stop_gradient(v_tm1)
    if normalise:
        adv_t_policy = safe_meta_gradient_standardise(adv_t_policy)
    policy_grad_loss = -jnp.mean(log_prob_tm1 * adv_t_policy)
    value_loss = jnp.mean((target_tm1 - v_tm1) ** 2)
    entropy_loss = jnp.mean(entropy)
    loss = (
        policy_grad_cost * policy_grad_loss
        - entropy_cost * entropy_loss
        + critic_cost * value_loss
    )
    metrics = {
        "loss": loss,
        "policy_grad_loss": policy_grad_loss,
        "entropy": entropy_loss,
        "entropy_step_0": jnp.mean(entropy[:, 0]),
        "value_loss": value_loss,
        "mean_value": jnp.mean(v_tm1),
        "mean_value_step0": jnp.mean(v_tm1[:, 0]),
        "mean_value_step1": jnp.mean(v_tm1[:, 1]),
        "mean_value_step-1": jnp.mean(v_tm1[:, -1]),
        "mean_value_step_middle": jnp.mean(v_tm1[:, v_tm1.shape[1] // 2]),
        "reward_step0": jnp.mean(r_t[:, 0]),
        "advantages_step0": jnp.mean(adv_t_policy[:, 0]),
        "advantages": jnp.mean(adv_t_policy),
    }
    return loss, metrics


def create_a2c_agent(
    env,
    forward_fn,
    true_value_fn: Optional[Callable] = None,
    n_step=10,
    batch_size_per_device=16,
    lambda_gae=0.95,
    entropy_cost=0.01,
    lr=5e-4,
    discount=0.98,
    policy_grad_cost=1.0,
    critic_cost=1.0,
    pmap_axis_name="num_devices",
    normalise: bool = False,
) -> OnlineAgent:
    optimizer = optax.adam(lr)

    def select_action(
        state: chex.ArrayTree, observation: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.ArrayTree, ExtrasA2C]:
        policy, value = forward_fn.apply(
            state,
            observation[None, ...],
        )
        action = jnp.squeeze(policy.sample(seed=key), axis=0)
        log_prob = policy.log_prob(action)
        entropy = policy.entropy()
        extras = ExtrasA2C(
            log_prob=jnp.squeeze(log_prob, axis=0),
            value=jnp.squeeze(value, axis=0),
            entropy=jnp.squeeze(entropy, axis=0),
            probs=jnp.squeeze(jax.nn.softmax(policy.logits, axis=-1)),
        )
        return action, extras

    data_generator = make_n_step_data_generator(
        select_action=select_action,
        environment=env,
        n_step=n_step,
        batch_size_per_device=batch_size_per_device,
    )

    def init(agent_key: chex.PRNGKey, data_generator_key: chex.PRNGKey):
        obs = env.spec.observations.generate_value()[None, ...]
        params = forward_fn.init(agent_key, obs)
        opt_state = optimizer.init(params)
        agent_state = AgentState(params=params, opt_state=opt_state)
        data_generator_state = data_generator.init(data_generator_key)
        state = State(agent_state, data_generator_state)
        return state

    def loss(
        params, data_generator_state
    ) -> Tuple[chex.Scalar, Tuple[DataGeneratorState, Metrics]]:
        (
            batch,
            data_generator_state,
            data_generation_info,
        ) = data_generator.generate_data(params, data_generator_state)
        if true_value_fn:
            obs = jnp.concatenate(
                [batch.observation, batch.next_observation[:, -1][:, None]], axis=1
            )
            probs = jnp.concatenate(
                [
                    batch.extras.probs,
                    jnp.zeros_like(batch.extras.probs[:, -1][:, None]),
                ],
                axis=1,
            )
            value = jax.vmap(
                jax.vmap(true_value_fn, in_axes=(0, 0, None)), in_axes=(0, 0, None)
            )(obs, probs, jax.lax.stop_gradient(discount))
            v_tm1 = value[:, :-1]
            v_t = value[:, 1:]
        else:
            v_tm1 = batch.extras.value
            _, v_t_bootstrap = forward_fn.apply(params, batch.observation[:, -1])
            v_t = jnp.concatenate(
                [batch.extras.value[:, 1:], v_t_bootstrap[:, None, ...]], axis=1
            )
        loss, metrics = a2c_loss_and_metrics(
            log_prob_tm1=batch.extras.log_prob,
            entropy=batch.extras.entropy,
            v_tm1=v_tm1,
            r_t=batch.reward,
            discount_t=batch.discount,
            v_t=v_t,
            lambda_gae=lambda_gae,
            discount=discount,
            entropy_cost=entropy_cost,
            policy_grad_cost=policy_grad_cost,
            critic_cost=critic_cost,
            normalise=normalise,
        )
        metrics.update(data_generation_info.metrics)
        if hasattr(batch.extras, "probs"):
            probs = jnp.mean(batch.extras.probs[:, 0], axis=0)
            metrics.update({"probs" + str(i): probs[i] for i in range(probs.shape[0])})
        return loss, (data_generator_state, metrics)

    def update(state: State) -> Tuple[State, Metrics]:
        grad, (data_generator_state, metrics) = jax.grad(loss, has_aux=True)(
            state.agent_state.params, state.data_generator_state
        )
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
        updates, opt_state = optimizer.update(grad, state.agent_state.opt_state)
        params = optax.apply_updates(state.agent_state.params, updates)
        agent_state = AgentState(params=params, opt_state=opt_state)
        state = State(
            agent_state=agent_state, data_generator_state=data_generator_state
        )
        return state, metrics

    return OnlineAgent(init=init, update=update, select_action=select_action)
