from typing import NamedTuple, Optional, Sequence

import omegaconf


class Config(NamedTuple):
    name: str
    agent: str
    meta_objective: Optional[str]
    outer_critic: bool
    num_timesteps: int
    num_eval_points: int
    total_batch_size: int
    total_num_envs: int
    n_steps: int
    total_num_eval: int
    deterministic_eval: bool
    network_type: str
    num_channels: int
    policy_layers: Sequence[int]
    value_layers: Sequence[int]
    embedding_size_actor: Optional[int]
    embedding_size_critic: Optional[int]
    optimizer: str
    learning_rate: float
    gradient_clip_norm: Optional[float]
    bootstrap_l_optimizer: Optional[str]
    bootstrap_l_learning_rate: Optional[float]
    meta_optimizer: Optional[str]
    meta_learning_rate: Optional[float]
    meta_gradient_clip_norm: Optional[float]
    normalize_advantage: bool
    normalize_outer_advantage: Optional[bool]
    reward_scaling: float
    gamma_init: float
    gamma_outer: Optional[float]
    gamma_metal: Optional[bool]
    lambda_init: float
    lambda_outer: Optional[float]
    lambda_metal: Optional[bool]
    l_pg_init: float
    l_pg_outer: Optional[float]
    l_pg_metal: Optional[bool]
    l_td_init: float
    l_td_outer: Optional[float]
    l_td_metal: Optional[bool]
    l_en_init: float
    l_en_outer: Optional[float]
    l_en_metal: Optional[bool]
    l_kl_outer: Optional[float]
    bootstrap_l: Optional[int]
    bootstrap_pm: Optional[float]
    bootstrap_vm: Optional[float]
    seed: int


def convert_config(cfg: omegaconf.DictConfig) -> Config:
    return Config(
        name=cfg.training.name,
        agent=cfg.agent.agent,
        meta_objective=cfg.agent.meta_objective,
        num_timesteps=cfg.training.num_timesteps,
        num_eval_points=cfg.training.num_eval_points,
        reward_scaling=cfg.training.reward_scaling,
        n_steps=cfg.training.n_steps,
        total_batch_size=cfg.training.total_batch_size,
        total_num_envs=cfg.training.total_num_envs,
        optimizer=cfg.training.optimizer,
        learning_rate=float(cfg.training.learning_rate),
        bootstrap_l_optimizer=cfg.agent.bootstrap_l_optimizer,
        bootstrap_l_learning_rate=cfg.agent.bootstrap_l_learning_rate,
        meta_optimizer=cfg.agent.meta_optimizer,
        meta_learning_rate=cfg.agent.meta_learning_rate,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        meta_gradient_clip_norm=cfg.agent.meta_gradient_clip_norm,
        gamma_init=float(cfg.training.gamma_init),
        gamma_outer=float(cfg.agent.gamma_outer)
        if cfg.agent.gamma_outer is not None
        else None,
        gamma_metal=cfg.agent.gamma_metal,
        lambda_init=float(cfg.training.lambda_init),
        lambda_outer=float(cfg.agent.lambda_outer)
        if cfg.agent.lambda_outer is not None
        else None,
        lambda_metal=cfg.agent.lambda_metal,
        l_pg_init=float(cfg.training.l_pg_init),
        l_pg_outer=float(cfg.agent.l_pg_outer)
        if cfg.agent.l_pg_outer is not None
        else None,
        l_pg_metal=cfg.agent.l_pg_metal,
        l_td_init=float(cfg.training.l_td_init),
        l_td_outer=float(cfg.agent.l_td_outer)
        if cfg.agent.l_td_outer is not None
        else None,
        l_td_metal=cfg.agent.l_td_metal,
        l_en_init=float(cfg.training.l_en_init),
        l_en_outer=float(cfg.agent.l_en_outer)
        if cfg.agent.l_en_outer is not None
        else None,
        l_en_metal=cfg.agent.l_en_metal,
        l_kl_outer=float(cfg.agent.l_kl_outer)
        if cfg.agent.l_kl_outer is not None
        else None,
        bootstrap_l=cfg.agent.bootstrap_l,
        bootstrap_pm=cfg.agent.bootstrap_pm,
        bootstrap_vm=cfg.agent.bootstrap_vm,
        normalize_advantage=cfg.agent.normalize_advantage,
        normalize_outer_advantage=cfg.agent.normalize_outer_advantage,
        total_num_eval=cfg.agent.total_num_eval,
        deterministic_eval=cfg.agent.deterministic_eval,
        policy_layers=cfg.network.policy_layers,
        value_layers=cfg.network.value_layers,
        outer_critic=cfg.agent.outer_critic,
        seed=cfg.training.seed,
        embedding_size_actor=cfg.network.embedding_size_actor,
        embedding_size_critic=cfg.network.embedding_size_critic,
        network_type=cfg.network.type,
        num_channels=cfg.network.num_channels,
    )
