from functools import partial

import jax.random
from acme.jax import utils as acme_utils
from jax import numpy as jnp
from tqdm import tqdm

from discounting_chain.base import OnlineAgent
from discounting_chain.list_logger import ListLogger


def outer_iter(state, update_fn, n_updates):
    state, metrics = jax.lax.scan(
        lambda state, xs: update_fn(state), init=state, xs=None, length=n_updates
    )
    metrics = jax.tree_map(lambda x: x[-1], metrics)  # last device, last element
    return state, metrics


def run(
    num_iterations: int,
    n_updates_per_iter: int,
    agent: OnlineAgent,
    logger: ListLogger,
    seed: int = 0,
):
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    devices = jax.devices()
    num_devices = len(devices)
    pmap_axis_name = "num_devices"
    outer_iter_fn = partial(
        outer_iter, update_fn=agent.update, n_updates=n_updates_per_iter
    )
    outer_iter_fn = jax.pmap(outer_iter_fn, axis_name=pmap_axis_name, devices=devices)
    state = jax.pmap(agent.init)(
        jnp.stack([key1] * num_devices), jax.random.split(key2, num_devices)
    )

    for _ in tqdm(range(num_iterations)):
        state, metrics = outer_iter_fn(state)
        metrics = acme_utils.get_from_first_device(metrics)
        logger.write(metrics)

    return acme_utils.get_from_first_device(state)
