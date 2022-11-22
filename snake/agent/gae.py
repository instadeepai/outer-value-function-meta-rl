import functools
from typing import Tuple

import chex
import jax
import rlax
from jax import numpy as jnp


def compute_td_lambda(
    discount: chex.Array,
    rewards: chex.Array,
    values: chex.Array,
    bootstrap_value: chex.Array,
    lambda_: float,
    discount_factor: float,
) -> Tuple[chex.Array, chex.Array]:
    v_tm1 = values
    v_t = jnp.concatenate([values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    r_t = rewards
    discount_t = discount * discount_factor
    advantages = jax.vmap(
        functools.partial(rlax.td_lambda, lambda_=lambda_, stop_target_gradients=False),
        in_axes=1,
        out_axes=1,
    )(
        v_tm1,
        r_t,
        discount_t,
        v_t,
    )
    vs = advantages + v_tm1
    return vs, advantages
