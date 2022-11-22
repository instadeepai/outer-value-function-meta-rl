from typing import Callable, Dict, NamedTuple, Optional

import chex
import haiku as hk
import jumanji.types
import optax
from jax import numpy as jnp


class HyperParams(NamedTuple):
    gamma: jnp.float_
    lambda_: jnp.float_
    l_pg: jnp.float_  # policy gradient loss cost
    l_td: jnp.float_  # temporal difference loss cost
    l_en: jnp.float_  # entropy loss cost


MetaParams = HyperParams


class Metal(NamedTuple):
    gamma: Callable[[chex.Array], chex.Array]
    lambda_: Callable[[chex.Array], chex.Array]
    l_pg: Callable[[chex.Array], chex.Array]
    l_td: Callable[[chex.Array], chex.Array]
    l_en: Callable[[chex.Array], chex.Array]


class ActorCriticParams(NamedTuple):
    actor: hk.Params
    critic: hk.Params
    outer_critic: Optional[hk.Params]


class TrainingState(NamedTuple):
    """Contains training state for the learner."""

    params: ActorCriticParams
    meta_params: Optional[MetaParams]
    optimizer_state: optax.OptState
    meta_optimizer_state: Optional[optax.OptState]
    env_steps: jnp.int32


class ActingState(NamedTuple):
    """Container for data used during the acting in the environment."""

    env_state: jumanji.env.State
    timestep: jumanji.types.TimeStep
    acting_key: chex.PRNGKey


class Transition(NamedTuple):
    """Container for a transition."""

    observation: chex.Array
    action: chex.Array
    reward: jnp.float_
    discount: jnp.float_
    truncation: jnp.bool_
    next_observation: chex.Array
    extras: Dict


class State(NamedTuple):
    """Container for TrainingState and ActingState."""

    training_state: TrainingState
    acting_state: ActingState
