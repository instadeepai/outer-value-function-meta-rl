from typing import TypeVar

import jax

T = TypeVar("T")


def first_from_device(tree: T) -> T:
    return jax.tree_util.tree_map(lambda x: x[0], tree)  # type: ignore
