from typing import Tuple

import chex
import jax
from acme.types import Transition
from jax import numpy as jnp

from discounting_chain.base import (
    AgentState,
    DataGenerationInfo,
    DataGenerator,
    Environment,
    SelectAction,
)
from discounting_chain.data_generation.n_step import (
    NStepGeneratorState,
    make_run_n_step,
)


def make_n_step_data_generator(
    select_action: SelectAction,
    environment: Environment,
    n_step: int,
    batch_size_per_device: int,
) -> DataGenerator:
    """
    Create a data generator that runs batches of n-step interaction with an environment,
    using the `run_n_step` function.

    Args:
        select_action: Callable specifying action selection by the agent.
        environment: Environment container specifying the environment state initialisation and
            step functions.
        n_step: Numer of environment steps per batch.
        batch_size_per_device: Batch size that we vmap over in the `run_n_step` function.
            We typically run the `generate_data` method across multiple devices, in which case the
            total batch size will be equal to batch_size_per_device*num_devices.

    Returns:
        The n-step data generator.

    """

    def init(rng_key: chex.PRNGKey) -> NStepGeneratorState:
        """Initialise the data generation state."""
        env_rng_key, data_generator_rng_key = jax.random.split(rng_key)
        env_rng_key_batch = jax.random.split(env_rng_key, batch_size_per_device)
        environment_state, time_step = jax.vmap(environment.init)(env_rng_key_batch)

        data_generation_state = NStepGeneratorState(
            environment_state=environment_state,
            previous_timestep=time_step,
            rng_key=jax.random.split(data_generator_rng_key, batch_size_per_device),
        )
        return data_generation_state

    run_n_step_fn = make_run_n_step(n_step, select_action, environment.step)

    def generate_data(
        agent_state: AgentState, data_generator_state: NStepGeneratorState
    ) -> Tuple[Transition, NStepGeneratorState, DataGenerationInfo]:
        """Generate a batch of data using the `run_n_step_fn`."""
        batch, data_generator_state, metrics = run_n_step_fn(
            agent_state, data_generator_state
        )
        data_generation_info = DataGenerationInfo(
            metrics=metrics,
            num_episodes=metrics["num_episodes"],
            num_steps=jnp.array(batch_size_per_device * n_step, "int"),
        )
        return batch, data_generator_state, data_generation_info

    data_generator = DataGenerator(init, generate_data)
    return data_generator
