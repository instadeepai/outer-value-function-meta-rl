import abc
from typing import Any, Callable, Dict, Mapping, Optional

import chex
import numpy as np
from neptune import new as neptune


class Logger(abc.ABC):
    # copied from Acme
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def save_checkpoint(self, file_name: str) -> None:
        """Saves a checkpoint."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""

    def __enter__(self) -> "Logger":
        return self

    def __exit__(
        self, exc_type: Exception, exc_val: Exception, exc_tb: Exception
    ) -> None:
        self.close()


class NeptuneLogger(Logger):
    def __init__(
        self,
        config: Dict[str, Any],
        aggregation_fn: Callable[[chex.Array], chex.Array],
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        self.run = neptune.init(
            project="<YOUR_NEPTUNE_PROJECT_NAME>",  # Change the project name to your Neptune project
            name=config["name"] + (f"_seed_{seed}" if seed is not None else ""),
            **kwargs,
        )
        self.run["config"] = config
        self._t: int = 0
        del aggregation_fn  # Moved to logging mean and last sample instead.
        self.mean_fn = np.mean

        def downsample(x: np.ndarray) -> np.ndarray:
            return x[-1]  # type: ignore

        self.downsample_fn = downsample

    def write(  # noqa: CCR001
        self,
        data: Mapping[str, Any],
        label: str = "",
        timestep: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._t = timestep or self._t + 1
        prefix = label and f"{label}/"
        for key, metric in data.items():
            if np.ndim(metric) == 0:
                if not np.isnan(metric):
                    self.run[f"{prefix}/{key}"].log(
                        float(metric),
                        step=self._t,
                        wait=True,
                    )
            elif np.ndim(metric) == 1:
                metric_value_mean = self.mean_fn(metric)
                if not np.isnan(metric_value_mean):
                    self.run[f"{prefix}mean/{key}"].log(
                        metric_value_mean.item(),
                        step=self._t,
                        wait=True,
                    )
                metric_value_sample = self.downsample_fn(metric)
                if not np.isnan(metric_value_sample):
                    self.run[f"{prefix}sample/{key}"].log(
                        metric_value_sample.item(),
                        step=self._t,
                        wait=True,
                    )
            else:
                raise ValueError(
                    f"Expected metric to be 0 or 1 dimension, got {metric}."
                )

    def save_checkpoint(self, file_name: str) -> None:
        self.run[f"checkpoints/{file_name}"].upload(file_name)

    def close(self) -> None:
        self.run.stop()


class TerminalLogger(Logger):
    def __init__(
        self, aggregation_fn: Callable[[chex.Array], chex.Array], **kwargs: Any
    ):
        self.aggregation_fn = aggregation_fn
        print(">>> Terminal Logger")

    def write(
        self,
        data: Mapping[str, Any],
        label: str = "",
        timestep: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        gamma = data.get("gamma", None)
        return_ = data.get("episode_reward_stochastic_policy", None)
        if timestep is not None:
            print_str = f"\nTimestep {timestep:.2e}  >>>  "
        else:
            print_str = "\n"
        if return_ is not None:
            print_str += f"mean_return: {self.aggregation_fn(return_):.2f}   "
        if gamma is not None:
            print_str += f"discount_factor: {self.aggregation_fn(gamma):.5f}"
        print(print_str)

    def save_checkpoint(self, file_name: str) -> None:
        pass

    def close(self) -> None:
        pass


def make_logger_factory(
    logger: str,
    config_dict: Dict[str, Any],
    aggregation_behaviour: str = "mean",
    seed: Optional[int] = None,
) -> Callable[[], Logger]:
    if aggregation_behaviour == "mean":
        aggregation_fn = np.mean
    else:
        raise ValueError(
            f"aggregation_behaviour is expected to be 'mean', got {aggregation_behaviour} instead."
        )

    def make_logger() -> Logger:
        if logger == "neptune":
            return NeptuneLogger(config_dict, aggregation_fn, seed)
        elif logger == "terminal":
            return TerminalLogger(aggregation_fn)
        else:
            raise ValueError(
                f"expected logger in ['neptune', 'terminal'], got {logger}."
            )

    return make_logger
