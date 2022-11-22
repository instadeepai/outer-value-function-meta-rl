from typing import Callable, NamedTuple, Optional

import chex
import haiku as hk
from jax import numpy as jnp

from snake.networks.distribution import ParametricDistribution


class FeedForwardNetwork(NamedTuple):
    init: Callable[
        [chex.PRNGKey, chex.Array, Optional[jnp.float_]],
        hk.Params,
    ]
    apply: Callable[
        [hk.Params, chex.Array, Optional[jnp.float_]],
        chex.Array,
    ]


class ActorCriticNetworks(NamedTuple):
    """Defines the actor-critic networks, which outputs the logits of a policy, and a value given
    an observation.
    """

    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    outer_value_network: Optional[FeedForwardNetwork]
    parametric_action_distribution: ParametricDistribution
