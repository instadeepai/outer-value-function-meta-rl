from typing import Callable, NamedTuple, Tuple

import chex
import jax.random
from acme.types import specs
from dm_env import TimeStep
from gymnax.environments.bsuite.discounting_chain import DiscountingChain
from jax import numpy as jnp

from discounting_chain.base import Environment, Metrics
from discounting_chain.envs.env_utils import wrap_env_for_episode_metrics

Action = chex.ArrayTree


class State(NamedTuple):
    gymnax_state: chex.ArrayTree
    key: chex.PRNGKey


def create_dc_gmnax(mapping_seed=3) -> Tuple[Environment, Callable]:
    gymnax_env = DiscountingChain(mapping_seed=mapping_seed)
    env_params = gymnax_env.default_params

    def get_true_value_non_first_step(obs, probs, gamma):
        gamma = jax.lax.stop_gradient(gamma)
        context, time = obs
        time = time * env_params.max_steps_in_episode + 1  # unnormalise time
        reward_step = env_params.reward_timestep[jnp.int_(context)]
        time_till_reward = reward_step - time
        reward_already_recieved = time_till_reward < 0
        reward = gymnax_env.reward[jnp.int_(context)]
        value = gamma ** time_till_reward * reward
        value = jax.lax.select(reward_already_recieved, 0.0, value)
        return value

    def get_true_value_first_step_single_action(action, gamma):
        time = 0
        reward_step = env_params.reward_timestep[jnp.int_(action)]
        time_till_reward = reward_step - time
        reward_already_recieved = time_till_reward < 0
        reward = gymnax_env.reward[jnp.int_(action)]
        value = gamma ** time_till_reward * reward
        value = jax.lax.select(reward_already_recieved, 0.0, value)
        return value

    def get_true_value_first_step(obs, probs, gamma):
        values = jax.vmap(get_true_value_first_step_single_action, in_axes=(0, None))(
            jnp.arange(probs.shape[0]), gamma
        )
        chex.assert_equal_shape([probs, values])
        return jnp.sum(probs * values)

    def get_true_value(obs, probs, gamma):
        context, time = obs
        value = jax.lax.cond(
            time == 0,
            get_true_value_first_step,
            get_true_value_non_first_step,
            obs,
            probs,
            gamma,
        )
        return jax.lax.stop_gradient(value)

    def init(key: chex.PRNGKey):
        key, subkey = jax.random.split(key)
        obs, state = gymnax_env.reset(subkey, env_params)
        timestep = TimeStep(
            step_type=jnp.int_(0),
            reward=jnp.float32(0.0),
            discount=jnp.float32(0.0),
            observation=obs,
        )
        state = State(state, key)
        return state, timestep

    def auto_reset(state: State, timestep: TimeStep) -> Tuple[State, TimeStep]:
        key, subkey = jax.random.split(state.key)
        obs, state = gymnax_env.reset(subkey)
        # Replace observation with reset observation.
        timestep = timestep._replace(observation=obs)
        state = State(state, key)
        return state, timestep

    def step(state: State, action: Action) -> Tuple[State, TimeStep, Metrics]:
        key, subkey = jax.random.split(state.key)
        n_obs, n_state, reward, done, info = gymnax_env.step_env(
            subkey, state.gymnax_state, action, env_params
        )
        timestep = TimeStep(
            step_type=jax.lax.select(done, jnp.int_(2), jnp.int_(1)),
            reward=jnp.float32(reward),
            discount=jnp.float32(1 - done),
            observation=n_obs,
        )
        state = State(n_state, key)

        state, timestep = jax.lax.cond(
            timestep.last(),
            auto_reset,
            lambda new_state, timestep: (new_state, timestep),
            state,
            timestep,
        )
        metrics = info
        return state, timestep, metrics

    gymnax_obs_spec = gymnax_env.observation_space(env_params)
    gymnax_action_spec = gymnax_env.action_space(env_params)

    spec = specs.EnvironmentSpec(
        observations=specs.Array(
            shape=gymnax_obs_spec.shape, dtype=gymnax_obs_spec.dtype
        ),
        actions=specs.DiscreteArray(
            num_values=gymnax_action_spec.n,
        ),
        rewards=None,
        discounts=None,
    )

    env = Environment(step=step, init=init, spec=spec)
    env = wrap_env_for_episode_metrics(env)
    return env, get_true_value
