from typing import Callable, NamedTuple, Tuple

import chex
import jax
import typing_extensions
from acme.types import Transition
from dm_env import TimeStep
from jax import numpy as jnp

from discounting_chain.base import (
    AgentState,
    EnvironmentState,
    Metrics,
    SelectAction,
    StepEnvironment,
)


class NStepGeneratorState(NamedTuple):
    """Container used within the `run_n_step` function for managing objects related to the
    generation of data through acting in the environment."""

    environment_state: EnvironmentState
    previous_timestep: TimeStep
    rng_key: chex.PRNGKey


class AccumulateBatchMetricsFn(typing_extensions.Protocol):
    """Accumulate metrics over a batch of n-step transitions."""

    def __call__(self, batch: Transition, metrics: Metrics) -> Metrics:
        """
        The function for accumulating the metrics.

        Args:
            batch: Batch of transition data with leading axis of (batch_size, n_step).
            metrics: Batch of n_step metrics with leading axis of (batch_size, n_step).

        Returns:
            metrics: Relevant information for logging.
        """


def _run_n_step(
    n_step: int,
    select_action: SelectAction,
    step_environment: StepEnvironment,
    agent_state: AgentState,
    data_generator_state: NStepGeneratorState,
) -> Tuple[Transition, NStepGeneratorState, Metrics]:
    """
    Run n environment steps (unbatched).

    Args:
        n_step: Number of environment steps.
        select_action: Function for action selection by the agent.
        step_environment: Function for stepping the environment.
        agent_state: State of the agent.
        data_generator_state: State for managing the environment interaction.

    Returns:
        transition: N-steps of experience
        data_generator_state: Updated data generator state
        metrics: Relevant information for logging.

    This function may be called with vmap for batching (see `make_run_n_step`).

    """

    def run_one_step(
        data_generator_state: NStepGeneratorState,
    ) -> Tuple[NStepGeneratorState, Tuple[Transition, Metrics]]:
        """Run a single step of acting - for use with `jax.lax.scan`."""
        rng_key, rng_subkey = jax.random.split(data_generator_state.rng_key)
        action, extras = select_action(
            agent_state, data_generator_state.previous_timestep.observation, rng_subkey
        )
        next_env_state, next_timestep, metrics = step_environment(
            data_generator_state.environment_state, action
        )
        trajectory = Transition(
            observation=data_generator_state.previous_timestep.observation,
            action=action,
            discount=next_timestep.discount,
            reward=next_timestep.reward,
            next_observation=next_timestep.observation,
            extras=extras,
        )
        data_generator_state = NStepGeneratorState(
            environment_state=next_env_state,
            previous_timestep=next_timestep,
            rng_key=rng_key,
        )
        return data_generator_state, (trajectory, metrics)

    next_data_generator_state, (trajectory, metrics) = jax.lax.scan(
        lambda act_state, xs: run_one_step(act_state),
        init=data_generator_state,
        xs=None,
        length=n_step,
    )

    return trajectory, next_data_generator_state, metrics


def accumulate_batch_metrics(batch: Transition, metrics: Metrics) -> Metrics:
    """Accumulate metrics over the batch of n-steps.

    This function assumes that `metrics` uses NaNs to mask values, when an environment step does
    not result in a metric that should be recorded. For example, because we only want to log
    episode returns for completed episodes, values within the batch of n-steps are typically NaN
    for all transitions where the episode is not yet complete.
    """
    # We calculate the number of complete episodes using the discount factor.
    num_episodes = jnp.sum(batch.discount == 0)
    accumulated_metrics = {"num_episodes": num_episodes}

    # Record the max and mean (masking NaN values) for metrics returned by the environment.
    mean_metrics = jax.tree_map(jnp.nanmean, metrics)
    max_metrics = jax.tree_map(jnp.nanmax, metrics)
    accumulated_metrics.update(
        {key + "_mean": val for key, val in mean_metrics.items()}
    )
    accumulated_metrics.update({key + "_max": val for key, val in max_metrics.items()})
    return accumulated_metrics


def make_run_n_step(
    n_step: int,
    select_action: SelectAction,
    step_environment: StepEnvironment,
    accumulate_batch_metrics_fn: AccumulateBatchMetricsFn = accumulate_batch_metrics,
) -> Callable[
    [AgentState, NStepGeneratorState], Tuple[Transition, NStepGeneratorState, Metrics]
]:
    """
    Create a run-n-step function for running n environment steps in batches.

    Args:
        n_step: Number of environment steps.
        select_action: Function for action selection by the agent.
        step_environment: Function for stepping the environment.
        accumulate_batch_metrics_fn: Function for accumulating metrics over the batch of n-steps.

    Returns:
        run_n_step: Function for generating experience through batched n-step interaction with
            an environment.

    """

    def run_n_step(
        agent_state: AgentState, data_generator_state: NStepGeneratorState
    ) -> Tuple[Transition, NStepGeneratorState, Metrics]:
        """
        Run a batch of n environment steps.

        Args:
            agent_state: State of the agent.
            data_generator_state: state used for managing the environment interaction,
                with the leading axis of each node in the pytree representing the batch size.

        Returns:
            trajectory_batch: Batch of experience, with leading axis of [batch_size, n_step].
            data_generator_state: Updated data_generator_state.
            metrics: Metrics accumulated during the generation of experience.

        """
        trajectory_batch, data_generator_state, episode_metrics = jax.vmap(
            _run_n_step, in_axes=(None, None, None, None, 0)
        )(n_step, select_action, step_environment, agent_state, data_generator_state)
        metrics = accumulate_batch_metrics_fn(trajectory_batch, episode_metrics)
        return trajectory_batch, data_generator_state, metrics

    return run_n_step
