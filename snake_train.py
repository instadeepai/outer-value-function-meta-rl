import functools
import logging
import pickle
from typing import Dict, Tuple

import hydra
import jax
import omegaconf
from tqdm import trange

from snake.agent import A2C, meta_params_to_hyper_params
from snake.training import utils
from snake.training.config import convert_config
from snake.training.logger import make_logger_factory
from snake.training.setup_run import setup
from snake.training.types import State


@hydra.main(config_path="snake/configs", config_name="config.yaml")
def run(cfg: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    logging.info({"devices": jax.local_devices()})

    config = convert_config(cfg)

    agent, evaluator = setup(config)
    # Can be either 'neptune' or 'terminal'.
    make_logger = make_logger_factory(
        "terminal", config._asdict(), aggregation_behaviour="mean"
    )

    num_timesteps_per_learner_inner_update_step = (
        config.total_batch_size * config.n_steps
    )
    num_timesteps_per_epoch = config.num_timesteps // config.num_eval_points
    num_learner_steps_per_epoch = (
        num_timesteps_per_epoch // num_timesteps_per_learner_inner_update_step
    )

    state = agent.init(jax.random.PRNGKey(config.seed))

    @functools.partial(jax.pmap, axis_name="devices")
    def epoch_fn(state: State) -> Tuple[State, Dict]:
        state, metrics = jax.lax.scan(
            lambda s, _: agent.update(s), state, None, num_learner_steps_per_epoch
        )
        return state, metrics

    with make_logger() as logger, trange(
        config.num_eval_points + 1
    ) as epochs, jax.log_compiles():
        num_learner_updates = 0

        # Log first hyper_params
        if state.training_state.meta_params is not None:
            initial_hyper_params = meta_params_to_hyper_params(
                utils.first_from_device(state.training_state.meta_params)
            )
        else:
            assert isinstance(agent, A2C)
            initial_hyper_params = agent.hyper_params
        logger.write(initial_hyper_params._asdict(), "train/mean/", num_learner_updates)
        logger.write(
            initial_hyper_params._asdict(), "train/sample/", num_learner_updates
        )

        for _ in epochs:

            # Checkpoint
            with open(f"state_{num_learner_updates:.2e}.pickle", "wb") as file_:
                pickle.dump(utils.first_from_device(state), file_)
                logger.save_checkpoint(file_.name)

            # Validation
            metrics = evaluator.run_evaluation(state.training_state)
            logger.write(utils.first_from_device(metrics), "eval", num_learner_updates)

            # Training steps
            state, metrics = epoch_fn(state)
            num_learner_updates += num_learner_steps_per_epoch
            logger.write(utils.first_from_device(metrics), "train", num_learner_updates)


if __name__ == "__main__":
    run()
