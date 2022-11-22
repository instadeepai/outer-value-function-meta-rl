from typing import NamedTuple, Tuple

import chex
import jax
from dm_env import TimeStep
from jax import numpy as jnp

from discounting_chain.base import Action, Environment, Metrics


class EnvStateWithEpisodeMetrics(NamedTuple):
    """Container that extends an environment state to additionally keep track of the step count
    within episodes, and the cumulative reward."""

    original_env_state: chex.ArrayTree
    episode_step_count: jnp.int32
    cumulative_reward: jnp.float32


def wrap_env_for_episode_metrics(env: Environment) -> Environment:
    """Wrapper for environment that records episode length and return, and adds them to the
    environment's metrics. NaN values are used for episode length & return in the metrics for
    non-terminal timesteps."""

    def accumulate_return_and_step_count(
        cumulative_reward: jnp.float32, step_count: jnp.int32, timestep: TimeStep
    ) -> Tuple[chex.Array, chex.Array]:
        """Update the episode's step count and cumulative return, resetting them to 0 if the episode
        is completed."""
        step_count = jax.lax.select(
            timestep.last(), jnp.array(0, dtype=int), step_count + 1
        )
        cumulative_reward = jax.lax.select(
            timestep.last(),
            jnp.array(0.0, dtype=float),
            cumulative_reward + timestep.reward,
        )
        return step_count, cumulative_reward

    def get_episode_metrics(
        previous_state: EnvStateWithEpisodeMetrics, timestep: TimeStep
    ) -> Metrics:
        """Get the episode metrics for the current timestep, if an episode is complete we return the
        episode return and length, otherwise we return NaN's to indicate the episode is still
        underway."""
        episode_return = jax.lax.select(
            timestep.last(),
            previous_state.cumulative_reward + timestep.reward,
            jnp.nan,
        )
        episode_length = jax.lax.select(
            timestep.last(),
            jnp.array(previous_state.episode_step_count + 1, dtype=float),
            jnp.nan,
        )
        metrics = {"episode_return": episode_return, "episode_length": episode_length}
        return metrics

    def init(rng_key: chex.PRNGKey) -> Tuple[EnvStateWithEpisodeMetrics, TimeStep]:
        """Initialise the environment's state."""
        original_env_state, timestep = env.init(rng_key)
        env_state = EnvStateWithEpisodeMetrics(
            original_env_state=original_env_state,
            episode_step_count=jnp.int32(0),
            cumulative_reward=jnp.float32(0),
        )
        return env_state, timestep

    def step(
        state: EnvStateWithEpisodeMetrics, action: Action
    ) -> Tuple[EnvStateWithEpisodeMetrics, TimeStep, Metrics]:
        """Step the environment."""
        original_env_state, timestep, metrics = env.step(
            state.original_env_state, action
        )

        episode_metrics = get_episode_metrics(state, timestep)
        metrics.update(episode_metrics)

        episode_step_count, cumulative_reward = accumulate_return_and_step_count(
            state.cumulative_reward, state.episode_step_count, timestep
        )
        state = EnvStateWithEpisodeMetrics(
            original_env_state=original_env_state,
            cumulative_reward=cumulative_reward,
            episode_step_count=episode_step_count,
        )

        return state, timestep, metrics

    return Environment(init=init, step=step, spec=env.spec)
