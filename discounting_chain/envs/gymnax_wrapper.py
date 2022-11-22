from typing import NamedTuple, Tuple

import chex
import gymnax
import jax.random
from acme.types import specs
from dm_env import TimeStep
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import numpy as jnp

from discounting_chain.base import Environment, Metrics
from discounting_chain.envs.env_utils import wrap_env_for_episode_metrics

Action = chex.ArrayTree


class State(NamedTuple):
    gymnax_state: chex.ArrayTree
    key: chex.PRNGKey


def create_gymnax_env(env_name="CartPole-v1", **kwargs) -> Environment:
    gymnax_env: GymnaxEnv
    if env_name == "DiscountingChain-bsuite":
        from gymnax.environments.bsuite.discounting_chain import DiscountingChain

        print(kwargs)
        gymnax_env = DiscountingChain(**kwargs)
        env_params = gymnax_env.default_params
    elif env_name == "UmbrellaChain-bsuite":
        from gymnax.environments.bsuite.umbrella_chain import EnvParams, UmbrellaChain

        gymnax_env = UmbrellaChain(n_distractor=0)
        change_length = 20 if not kwargs else kwargs["chain_length"]
        env_params = EnvParams(change_length, change_length)
    else:
        gymnax_env, env_params = gymnax.make(env_name)

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
    return env
