from typing import Callable, NamedTuple, Optional, Tuple

import chex
import jax.random
import optax
import rlax
from acme.types import Transition
from jax import numpy as jnp

from discounting_chain.a2c import (
    ExtrasA2C,
    a2c_loss_and_metrics,
    safe_meta_gradient_standardise,
)
from discounting_chain.base import DataGeneratorState, Metrics, OnlineAgent
from discounting_chain.data_generation.n_step_data_generator import (
    make_n_step_data_generator,
)
from discounting_chain.meta_a2c import (
    AgentState,
    MetaParams,
    MetaState,
    State,
    bijectors,
    setup_meta_to_hyper_params_fn,
)


class DoubleOptState(NamedTuple):
    inner: optax.OptState
    outer: optax.OptState


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
    use_first_crit_for_policy: bool = True,
    normalise: bool = True,
) -> Tuple[chex.Scalar, Metrics]:
    d_t = discount_t * discount
    d_t_outer = discount_t * discount_outer

    target_tm1 = jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None))(
        r_t, d_t, jax.lax.stop_gradient(v_t), lambda_gae
    )
    target_tm1_outer = jax.vmap(rlax.lambda_returns, in_axes=(0, 0, 0, None))(
        r_t, d_t_outer, jax.lax.stop_gradient(v_t_outer), lambda_gae
    )
    if use_first_crit_for_policy:
        adv_t_policy = target_tm1 - jax.lax.stop_gradient(v_tm1)
    else:
        adv_t_policy = target_tm1_outer - jax.lax.stop_gradient(v_tm1_outer)

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
    value_loss_outer = jnp.mean((target_tm1_outer - v_tm1_outer) ** 2)
    metrics.update(
        value_loss_crit2=value_loss_outer,
        mean_value_crit2=jnp.mean(v_tm1_outer),
        mean_value_crit2_step1=jnp.mean(v_tm1[:, 1]),
        mean_value_crit2_step0=jnp.mean(v_tm1[:, 0]),
        mean_value_step_last=jnp.mean(v_tm1[:, -1]),
    )
    loss = loss + value_loss_outer
    return loss, metrics


def create_bmg_a2c_agent(
    env,
    forward_fn,
    true_value_fn: Optional[Callable] = None,
    n_step=5,
    batch_size_per_device=64,
    lr=1e-4,
    meta_lr=1e-2,
    pmap_axis_name="num_devices",
    outer_discount=1.0,
    outer_lambda_gae=0.0,
    outer_entropy_cost=0.0,
    outer_policy_grad_cost=1.0,
    outer_critic_cost=1.0,
    init_discount=0.95,
    lambda_gae_inner=0.0,
    entropy_cost_inner=0.01,
    policy_grad_cost_inner=1.0,
    critic_cost_inner=1.0,
    meta_value_head: bool = False,
    only_bmg_updates: bool = False,
    n_bootstrap_target_updates: int = 3,
    normalise: bool = False,
    sgd_optimizer: bool = False,
    sgd_meta_optimizer: bool = False,
    kl_over_full_batch: bool = True,
    use_outer_optimizer=False,
) -> OnlineAgent:
    if sgd_optimizer:
        optimizer = optax.sgd(lr)
    else:
        optimizer = optax.rmsprop(lr)
        if use_outer_optimizer:
            outer_optimizer = optax.rmsprop(lr)

    if sgd_meta_optimizer:
        meta_optimizer = optax.sgd(meta_lr)
    else:
        meta_optimizer = optax.adam(meta_lr)

    meta_params_to_hyper_params = setup_meta_to_hyper_params_fn(
        lambda_gae=lambda_gae_inner,
        entropy_cost=entropy_cost_inner,
        policy_grad_cost=policy_grad_cost_inner,
        critic_cost=critic_cost_inner,
    )

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
        if use_outer_optimizer:
            inner_opt_state = optimizer.init(params)
            outer_opt_state = outer_optimizer.init(params)
            opt_state = DoubleOptState(inner_opt_state, outer_opt_state)
        else:
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
        bootstrap_target_update: bool = False,
    ) -> Tuple[chex.Scalar, Tuple[DataGeneratorState, Transition, Metrics]]:
        (
            batch,
            data_generator_state,
            data_generation_info,
        ) = data_generator.generate_data(params, data_generator_state)
        inner_discount = discount
        discount = outer_discount if bootstrap_target_update else discount
        lambda_gae = outer_lambda_gae if bootstrap_target_update else lambda_gae
        entropy_cost = outer_entropy_cost if bootstrap_target_update else entropy_cost
        critic_cost = outer_critic_cost if bootstrap_target_update else critic_cost
        policy_grad_cost = (
            outer_policy_grad_cost if bootstrap_target_update else policy_grad_cost
        )

        if meta_value_head:
            if true_value_fn:
                obs = jnp.concatenate(
                    [batch.observation, batch.next_observation[:, -1][:, None]], axis=1
                )
                # note: final action logits don't matter for value function, hence zeros
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
                value_outer = jax.vmap(
                    jax.vmap(true_value_fn, in_axes=(0, 0, None)), in_axes=(0, 0, None)
                )(obs, probs, jax.lax.stop_gradient(outer_discount))
                v_tm1 = value[:, :-1]
                v_t = value[:, 1:]
                v_tm1_outer = value_outer[:, :-1]
                v_t_outer = value_outer[:, 1:]
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
                use_first_crit_for_policy=not bootstrap_target_update,
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
        if hasattr(batch.extras, "probs"):
            probs = jnp.mean(batch.extras.probs[:, 0], axis=0)
            metrics.update({"probs" + str(i): probs[i] for i in range(probs.shape[0])})
        metrics.update(data_generation_info.metrics)
        return loss, (data_generator_state, batch, metrics)

    def inner_update(
        agent_state, data_generator_state, hyper_params, bootstrap_update: bool = False
    ) -> Tuple[AgentState, DataGeneratorState, Transition, Metrics]:
        grad, (data_generator_state, batch, metrics) = jax.grad(
            inner_loss, has_aux=True, argnums=0
        )(
            agent_state.params,
            data_generator_state,
            hyper_params.discount,
            hyper_params.lambda_gae,
            hyper_params.entropy_cost,
            hyper_params.policy_grad_cost,
            hyper_params.critic_cost,
            bootstrap_update,
        )
        grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
        if bootstrap_update:
            opt_state = (
                agent_state.opt_state.outer
                if use_outer_optimizer
                else agent_state.opt_state
            )
        else:
            opt_state = (
                agent_state.opt_state.inner
                if use_outer_optimizer
                else agent_state.opt_state
            )
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(agent_state.params, updates)
        if use_outer_optimizer:
            if bootstrap_update:
                opt_state = DoubleOptState(agent_state.opt_state.inner, opt_state)
            else:
                opt_state = DoubleOptState(opt_state, agent_state.opt_state.outer)
        agent_state = AgentState(params=params, opt_state=opt_state)
        metrics.update(discount_factor=hyper_params.discount)
        return agent_state, data_generator_state, batch, metrics

    def kl(params_target, params, observation):
        target_policy, _ = forward_fn.apply(params_target, observation)
        policy, _ = forward_fn.apply(params, observation)
        forward_kl = target_policy.kl_divergence(policy)
        return forward_kl

    def outer_loss(
        meta_params, agent_state, data_generator_state
    ) -> Tuple[chex.Scalar, Tuple[AgentState, DataGeneratorState, Metrics]]:
        hyper_params = meta_params_to_hyper_params(meta_params)
        agent_state, data_generator_state, _, metrics = inner_update(
            agent_state, data_generator_state, hyper_params, bootstrap_update=False
        )
        target_state = jax.lax.stop_gradient(agent_state)
        for i in range(n_bootstrap_target_updates):
            bootstrap_target = (
                i == (n_bootstrap_target_updates - 1)
            ) or only_bmg_updates
            target_state, data_generator_state, batch, meta_metrics = inner_update(
                target_state,
                data_generator_state,
                hyper_params,
                bootstrap_update=bootstrap_target,
            )

        target_state = jax.lax.stop_gradient(target_state)
        if kl_over_full_batch:
            forward_kl = jnp.mean(
                jax.vmap(kl, in_axes=(None, None, 0))(
                    target_state.params, agent_state.params, batch.observation
                )
            )
        else:
            forward_kl = jnp.mean(
                kl(target_state.params, agent_state.params, batch.observation[:, 0])
            )
        meta_loss = forward_kl
        if hasattr(batch.extras, "probs"):
            probs = jnp.mean(batch.extras.probs[:, 0], axis=0)
            metrics.update(
                {"probs_1_update" + str(i): probs[i] for i in range(probs.shape[0])}
            )

        if use_outer_optimizer:
            agent_state = agent_state._replace(opt_state=target_state.opt_state)
        metrics.update({"meta_" + key: value for key, value in meta_metrics.items()})
        metrics.update(meta_loss=meta_loss)
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
