from typing import Any, Callable, NamedTuple, Optional, Tuple

import chex
import jax.random
import optax
import rlax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from discounting_chain.a2c import (
    ExtrasA2C,
    a2c_loss_and_metrics,
    safe_meta_gradient_standardise,
)
from discounting_chain.base import DataGeneratorState, Metrics, OnlineAgent
from discounting_chain.data_generation.n_step_data_generator import (
    make_n_step_data_generator,
)


def double_head_a2c_loss_and_metrics(
    log_prob_tm1,
    entropy,
    v_tm1,
    v_tm1_outer,
    r_t,
    discount_t,
    v_t,
    v_t_outer,
    discount,
    discount_outer,
    lambda_gae: float,
    entropy_cost: float,
    policy_grad_cost: float,
    critic_cost: float,
    normalise: bool = True,
) -> Tuple[chex.Scalar, Metrics]:

    discount_t_outer = discount_t * discount_outer
    discount_t = discount_t * discount

    target_tm1 = jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None))(
        r_t, discount_t, jax.lax.stop_gradient(v_t), lambda_gae
    )
    adv_t_policy = target_tm1 - jax.lax.stop_gradient(v_tm1)
    if normalise:
        adv_t_policy = safe_meta_gradient_standardise(adv_t_policy)
    policy_grad_loss = jnp.mean(-log_prob_tm1 * adv_t_policy)
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
    stop_gradient = False

    td_lambda_crit2 = jax.vmap(rlax.td_lambda, in_axes=(0, 0, 0, 0, None, None))(
        v_tm1_outer,
        r_t,
        discount_t_outer,
        jax.lax.stop_gradient(v_t_outer),
        lambda_gae,
        stop_gradient,
    )
    value_loss_outer = jnp.mean(td_lambda_crit2 ** 2)
    metrics.update(
        value_loss_crit2=value_loss_outer, mean_value_crit2=jnp.mean(v_tm1_outer)
    )
    loss = loss + value_loss_outer
    return loss, metrics


class MetaParams(NamedTuple):
    discount: chex.Scalar


class HyperParams(NamedTuple):
    discount: chex.Scalar
    lambda_gae: chex.Scalar
    entropy_cost: chex.Scalar
    policy_grad_cost: chex.Scalar
    critic_cost: chex.Scalar


class Bijectors(NamedTuple):
    discount: Any = tfp.bijectors.Sigmoid(low=0.90, high=1.0)


bijectors = Bijectors()


class AgentState(NamedTuple):
    params: chex.ArrayTree
    opt_state: optax.OptState


class MetaState(NamedTuple):
    params: MetaParams
    opt_state: optax.OptState


class State(NamedTuple):
    agent_state: AgentState
    meta_state: MetaState
    data_generator_state: DataGeneratorState


def setup_meta_to_hyper_params_fn(
    lambda_gae, entropy_cost, policy_grad_cost, critic_cost
):
    def meta_params_to_hyper_params(
        meta_params: MetaParams,
    ) -> HyperParams:
        discount = bijectors.discount.forward(meta_params.discount)
        hyper_params = HyperParams(
            discount=discount,
            lambda_gae=lambda_gae,
            entropy_cost=entropy_cost,
            policy_grad_cost=policy_grad_cost,
            critic_cost=critic_cost,
        )
        return hyper_params

    return meta_params_to_hyper_params


def create_meta_a2c_agent(
    env,
    forward_fn,
    true_value_fn: Optional[Callable] = None,
    n_step=5,
    batch_size_per_device=64,
    lr=3e-4,
    meta_lr=1e-3,
    pmap_axis_name="num_devices",
    outer_discount=1.0,
    outer_lambda_gae=0.0,
    outer_entropy_cost=0.0,
    outer_policy_grad_cost=1.0,
    outer_critic_cost=0.0,
    init_discount=0.9,
    lambda_gae=0.0,
    entropy_cost=0.01,
    policy_grad_cost=1.0,
    critic_cost=1.0,
    meta_value_head: bool = False,
    sgd_optimizer: bool = False,
    sgd_meta_optimizer: bool = False,
    normalise=False,
) -> OnlineAgent:
    if sgd_optimizer:
        optimizer = optax.sgd(lr)
    else:
        optimizer = optax.rmsprop(lr)

    if sgd_meta_optimizer:
        meta_optimizer = optax.sgd(meta_lr)
    else:
        meta_optimizer = optax.adam(meta_lr)

    meta_params_to_hyper_params = setup_meta_to_hyper_params_fn(
        lambda_gae=lambda_gae,
        entropy_cost=entropy_cost,
        policy_grad_cost=policy_grad_cost,
        critic_cost=critic_cost,
    )

    def select_action(
        params: chex.ArrayTree, observation: chex.Array, key: chex.PRNGKey
    ) -> Tuple[chex.ArrayTree, ExtrasA2C]:

        policy, value = forward_fn.apply(
            params,
            observation[None, ...],
        )
        action = jnp.squeeze(policy.sample(seed=key), axis=0)
        log_prob = policy.log_prob(action)
        entropy = policy.entropy()
        extras = ExtrasA2C(
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            probs=jax.nn.softmax(policy.logits, axis=-1),
        )
        extras = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), extras)
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
        meta_params = MetaParams(discount=bijectors.discount.inverse(init_discount))
        meta_opt_state = meta_optimizer.init(meta_params)
        meta_state = MetaState(params=meta_params, opt_state=meta_opt_state)
        data_generator_state = data_generator.init(data_generator_key)
        state = State(
            agent_state=agent_state,
            meta_state=meta_state,
            data_generator_state=data_generator_state,
        )
        return state

    def inner_loss(
        params,
        data_generator_state,
        discount,
        lambda_gae,
        entropy_cost,
        policy_grad_cost,
        critic_cost,
    ) -> Tuple[chex.Scalar, Tuple[DataGeneratorState, Metrics]]:
        (
            batch,
            data_generator_state,
            data_generation_info,
        ) = data_generator.generate_data(params, data_generator_state)
        inner_discount = discount
        if meta_value_head:
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
                )(obs, probs, jax.lax.stop_gradient(inner_discount))
                v_tm1 = value[:, :-1]
                v_t = value[:, 1:]
                v_tm1_outer = jnp.zeros_like(v_tm1)  # not used
                v_t_outer = jnp.zeros_like(v_tm1)  # not used
            else:
                v_tm1 = batch.extras.value[0]
                v_tm1_outer = batch.extras.value[1]

                _, (v_t_bootstrap, v_t_bootstrap_outer) = forward_fn.apply(
                    params, batch.next_observation[:, -1]
                )
                v_t = jnp.concatenate(
                    [batch.extras.value[0][:, 1:], v_t_bootstrap[:, None, ...]], axis=1
                )
                v_t_outer = jnp.concatenate(
                    [batch.extras.value[1][:, 1:], v_t_bootstrap_outer[:, None, ...]],
                    axis=1,
                )

            loss, metrics = double_head_a2c_loss_and_metrics(
                log_prob_tm1=batch.extras.log_prob,
                entropy=batch.extras.entropy,
                v_tm1=v_tm1,
                v_tm1_outer=v_tm1_outer,
                r_t=batch.reward,
                discount_t=batch.discount,
                v_t=v_t,
                v_t_outer=v_t_outer,
                lambda_gae=lambda_gae,
                discount=discount,
                discount_outer=outer_discount,
                entropy_cost=entropy_cost,
                policy_grad_cost=policy_grad_cost,
                critic_cost=critic_cost,
                normalise=normalise,
            )
        else:
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
                )(obs, probs, jax.lax.stop_gradient(inner_discount))
                v_tm1 = value[:, :-1]
                v_t = value[:, 1:]
            else:
                _, v_t_bootstrap = forward_fn.apply(params, batch.observation[:, -1])
                v_t = jnp.concatenate(
                    [batch.extras.value[:, 1:], v_t_bootstrap[:, None, ...]], axis=1
                )
                v_tm1 = batch.extras.value
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

    def inner_update(agent_state, data_generator_state, hyper_params):
        grad, (data_generator_state, metrics) = jax.grad(
            inner_loss, has_aux=True, argnums=0
        )(
            agent_state.params,
            data_generator_state,
            hyper_params.discount,
            hyper_params.lambda_gae,
            hyper_params.entropy_cost,
            hyper_params.policy_grad_cost,
            hyper_params.critic_cost,
        )
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
        updates, opt_state = optimizer.update(grad, agent_state.opt_state)
        params = optax.apply_updates(agent_state.params, updates)
        agent_state = AgentState(params=params, opt_state=opt_state)
        return agent_state, data_generator_state, metrics

    def outer_loss(
        meta_params, agent_state, data_generator_state
    ) -> Tuple[chex.Scalar, Tuple[AgentState, DataGeneratorState, Metrics]]:
        hyper_params = meta_params_to_hyper_params(meta_params)
        agent_state, data_generator_state, metrics = inner_update(
            agent_state, data_generator_state, hyper_params
        )
        batch, data_generator_state, _ = data_generator.generate_data(
            agent_state.params, data_generator_state
        )
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
            )(
                obs,
                probs,
                outer_discount
                if meta_value_head
                else jax.lax.stop_gradient(hyper_params.discount),
            )
            v_tm1 = value[:, :-1]
            v_t = value[:, 1:]
        else:
            if meta_value_head:
                v_tm1 = batch.extras.value[1]
                _, (_, v_t_bootstrap) = forward_fn.apply(
                    agent_state.params, batch.next_observation[:, -1]
                )
            else:
                v_tm1 = batch.extras.value
                _, v_t_bootstrap = forward_fn.apply(
                    agent_state.params, batch.next_observation[:, -1]
                )
            v_t = jnp.concatenate([v_tm1[:, 1:], v_t_bootstrap[:, None]], axis=1)
        meta_loss, meta_metrics = a2c_loss_and_metrics(
            log_prob_tm1=batch.extras.log_prob,
            entropy=batch.extras.entropy,
            v_tm1=v_tm1,
            r_t=batch.reward,
            discount_t=batch.discount,
            v_t=v_t,
            discount=outer_discount,
            lambda_gae=outer_lambda_gae,
            entropy_cost=outer_entropy_cost,
            policy_grad_cost=outer_policy_grad_cost,
            critic_cost=outer_critic_cost,
            normalise=normalise,
        )
        if hasattr(batch.extras, "probs"):
            probs = jnp.mean(batch.extras.probs[:, 0], axis=0)
            meta_metrics.update(
                {"probs" + str(i): probs[i] for i in range(probs.shape[0])}
            )
        metrics.update({"meta_" + key: value for key, value in meta_metrics.items()})
        metrics.update(discount_factor=hyper_params.discount)
        return meta_loss, (agent_state, data_generator_state, metrics)

    def update(state: State) -> Tuple[State, Metrics]:
        grad, (agent_state, data_generator_state, metrics) = jax.grad(
            outer_loss, has_aux=True, argnums=0
        )(state.meta_state.params, state.agent_state, state.data_generator_state)
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
        updates, opt_state = meta_optimizer.update(grad, state.meta_state.opt_state)
        metrics.update(
            meta_grad=grad.discount,
            meta_update=updates.discount,
        )
        meta_params = optax.apply_updates(state.meta_state.params, updates)
        meta_state = MetaState(params=meta_params, opt_state=opt_state)
        state = State(
            agent_state=agent_state,
            data_generator_state=data_generator_state,
            meta_state=meta_state,
        )
        return state, metrics

    return OnlineAgent(init=init, update=update, select_action=select_action)
