import chex
import jax
from acme.specs import EnvironmentSpec
from acme.types import Transition
from jax import numpy as jnp


def create_fake_n_step_batch(
    environment_spec: EnvironmentSpec,
    batch_size: int,
    n_step: int,
    fake_extras: chex.ArrayTree = (),
) -> Transition:
    """
    Create a fake batch of data that matches what is returned by the n-step generator.
    This is useful for initialisation of the buffer, which has a state initialisation that
    depends on the batch it receives from experience generation.

    Args:
        environment_spec: Environment spec. Each field in the `environment_spec` has to implement
            the `generate_value` method for this function to work.
        fake_extras: An object matching the shapes and types of Extras returned by the
            `select_action` method of an agent.
    Returns:
        batch: Batch of data matching what the n-step generator returns.
    """
    transition = Transition(
        observation=environment_spec.observations.generate_value(),
        action=environment_spec.actions.generate_value(),
        reward=environment_spec.rewards.generate_value(),
        next_observation=environment_spec.observations.generate_value(),
        discount=environment_spec.discounts.generate_value(),
        extras=fake_extras,
    )
    batch = jax.tree_map(
        lambda x: jnp.broadcast_to(x, shape=(batch_size, n_step, *x.shape)), transition
    )
    return batch
