from typing import Tuple

import jax
import jumanji
import optax

from snake.agent import A2C, ActorCriticAgent, MetaA2C
from snake.env import make_snake_env
from snake.networks import make_actor_critic_networks_snake
from snake.networks.actor_critic import ActorCriticNetworks
from snake.training.config import Config
from snake.training.evaluator import Evaluator
from snake.training.types import HyperParams, Metal


def setup(config: Config) -> Tuple[ActorCriticAgent, Evaluator]:
    env, eval_env = make_snake_env()

    agent = make_agent(config, env)
    evaluator = Evaluator(
        eval_env=eval_env,
        actor_critic_agent=agent,
        total_num_eval=config.total_num_eval,
        key=jax.random.PRNGKey(config.seed),
        deterministic=config.deterministic_eval,
    )
    return agent, evaluator


def make_agent(  # noqa: CCR001
    config: Config, env: jumanji.Environment
) -> ActorCriticAgent:

    actor_critic_networks: ActorCriticNetworks
    if config.network_type == "snake":
        actor_critic_networks = make_actor_critic_networks_snake(
            num_channels=config.num_channels,
            policy_layers=config.policy_layers,
            value_layers=config.value_layers,
            outer_critic=config.outer_critic,
            embedding_size_actor=config.embedding_size_actor,
            embedding_size_critic=config.embedding_size_critic,
        )
    else:
        raise ValueError

    if config.optimizer == "sgd":
        optimizer = optax.sgd(config.learning_rate)
    elif config.optimizer == "rmsprop":
        optimizer = optax.rmsprop(config.learning_rate, decay=0.9)
    elif config.optimizer == "adam":
        optimizer = optax.adam(config.learning_rate, eps_root=1e-8)
    else:
        raise ValueError(
            f"Expected optimizer to be in ['sgd', 'rmsprop', 'adam'], got "
            f"{config.optimizer} instead."
        )

    if config.gradient_clip_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip_norm),
            optimizer,
        )
    agent: ActorCriticAgent
    if config.agent == "a2c":
        agent = A2C(
            n_steps=config.n_steps,
            total_batch_size=config.total_batch_size,
            total_num_envs=config.total_num_envs,
            env=env,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=config.normalize_advantage,
            reward_scaling=config.reward_scaling,
            hyper_params=HyperParams(
                gamma=config.gamma_init,
                lambda_=config.lambda_init,
                l_pg=config.l_pg_init,
                l_td=config.l_td_init,
                l_en=config.l_en_init,
            ),
            env_type=config.network_type,
        )
    elif config.agent == "meta_a2c":
        assert config.meta_learning_rate is not None
        if config.meta_optimizer == "sgd":
            meta_optimizer = optax.sgd(config.meta_learning_rate)
        elif config.meta_optimizer == "rmsprop":
            meta_optimizer = optax.rmsprop(config.meta_learning_rate, decay=0.9)
        elif config.meta_optimizer == "adam":
            meta_optimizer = optax.adam(config.meta_learning_rate, eps_root=1e-8)
        else:
            raise ValueError(
                f"Expected meta_optimizer to be in ['sgd', 'rmsprop', 'adam'], got "
                f"{config.meta_optimizer} instead."
            )
        if config.meta_gradient_clip_norm is not None:
            meta_optimizer = optax.chain(
                optax.clip_by_global_norm(config.meta_gradient_clip_norm),
                meta_optimizer,
            )

        if config.bootstrap_l_optimizer is not None:
            if config.bootstrap_l_optimizer == "sgd":
                bootstrap_l_optimizer = optax.sgd(config.bootstrap_l_learning_rate)
            elif config.bootstrap_l_optimizer == "rmsprop":
                bootstrap_l_optimizer = optax.rmsprop(
                    config.bootstrap_l_learning_rate, decay=0.9
                )
            elif config.bootstrap_l_optimizer == "adam":
                bootstrap_l_optimizer = optax.adam(
                    config.bootstrap_l_learning_rate, eps_root=1e-8
                )
            else:
                raise ValueError(
                    f"Expected bootstrap_l_optimizer to be in ['sgd', 'rmsprop', 'adam'], got "
                    f"{config.bootstrap_l_optimizer} instead."
                )
        else:
            bootstrap_l_optimizer = None

        hyper_params_init = HyperParams(
            gamma=config.gamma_init,
            lambda_=config.lambda_init,
            l_pg=config.l_pg_init,
            l_td=config.l_td_init,
            l_en=config.l_en_init,
        )
        outer_hyper_params = HyperParams(
            gamma=config.gamma_outer,
            lambda_=config.lambda_outer,
            l_pg=config.l_pg_outer,
            l_td=config.l_td_outer,
            l_en=config.l_en_outer,
        )
        assert config.normalize_outer_advantage is not None
        metal = Metal(
            gamma=(lambda _: _) if config.gamma_metal else jax.lax.stop_gradient,
            lambda_=(lambda _: _) if config.lambda_metal else jax.lax.stop_gradient,
            l_pg=(lambda _: _) if config.l_pg_metal else jax.lax.stop_gradient,
            l_td=(lambda _: _) if config.l_td_metal else jax.lax.stop_gradient,
            l_en=(lambda _: _) if config.l_en_metal else jax.lax.stop_gradient,
        )
        agent = MetaA2C(
            n_steps=config.n_steps,
            total_batch_size=config.total_batch_size,
            total_num_envs=config.total_num_envs,
            env=env,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            bootstrap_l_optimizer=bootstrap_l_optimizer,
            meta_optimizer=meta_optimizer,
            normalize_advantage=config.normalize_advantage,
            normalize_outer_advantage=config.normalize_outer_advantage,
            reward_scaling=config.reward_scaling,
            hyper_params_init=hyper_params_init,
            outer_hyper_params=outer_hyper_params,
            meta_objective=config.meta_objective,
            bootstrap_l=config.bootstrap_l,
            bootstrap_pm=config.bootstrap_pm,
            bootstrap_vm=config.bootstrap_vm,
            metal=metal,
            l_kl_outer=config.l_kl_outer,
            env_type=config.network_type,
        )
    else:
        raise ValueError(
            f"Expected agent in ['a2c', 'meta_a2c'], got {config.agent} instead."
        )
    return agent
