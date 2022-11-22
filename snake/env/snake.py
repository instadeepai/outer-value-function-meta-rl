from typing import Tuple

import jumanji
import jumanji.wrappers


def make_snake_env() -> Tuple[jumanji.Environment, jumanji.Environment]:
    snake = jumanji.make("Snake-6x6-v0")
    eval_snake = snake
    snake = jumanji.wrappers.AutoResetWrapper(snake)
    snake = jumanji.wrappers.VmapWrapper(snake)
    return snake, eval_snake
