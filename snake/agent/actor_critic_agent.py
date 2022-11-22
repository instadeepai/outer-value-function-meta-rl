import abc
from typing import Callable, Dict, Optional, Tuple

import chex
import haiku as hk
import jax.random
import jumanji
import optax
from jax import numpy as jnp

from snake.networks.actor_critic import ActorCriticNetworks
from snake.training.types import ActingState, ActorCriticParams, State, Transition


class ActorCriticAgent(abc.ABC):
    def __init__(
        self,
        n_steps: int,
        total_batch_size: int,
        total_num_envs: int,
        env: jumanji.Environment,
        actor_critic_networks: ActorCriticNetworks,
        optimizer: optax.GradientTransformation,
        normalize_advantage: bool,
        reward_scaling: float,
        env_type: str,
    ):
        self._n_steps = n_steps
        num_devices = jax.local_device_count()

        self._total_batch_size = total_batch_size
        assert total_batch_size % num_devices == 0
        self._batch_size_per_device: int = total_batch_size // num_devices

        self._total_num_envs = total_num_envs
        assert total_num_envs % num_devices == 0
        self._num_envs_per_device: int = total_num_envs // num_devices

        self._env = env
        self._actor_critic_networks = actor_critic_networks
        self._optimizer = optimizer
        self._normalize_advantage = normalize_advantage
        self._reward_scaling = reward_scaling
        self._env_type = env_type

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def total_batch_size(self) -> int:
        return self._total_batch_size

    @property
    def batch_size_per_device(self) -> int:
        return self._batch_size_per_device

    @property
    def num_envs_per_device(self) -> int:
        return self._num_envs_per_device

    @property
    def env(self) -> jumanji.Environment:
        return self._env

    @property
    def actor_critic_networks(self) -> ActorCriticNetworks:
        return self._actor_critic_networks

    @property
    def optimizer(self) -> optax.GradientTransformation:
        return self._optimizer

    @property
    def normalize_advantage(self) -> bool:
        return self._normalize_advantage

    @property
    def reward_scaling(self) -> float:
        return self._reward_scaling

    @property
    def env_type(self) -> str:
        return self._env_type

    @abc.abstractmethod
    def init(self, key: chex.PRNGKey) -> State:
        pass

    @abc.abstractmethod
    def update(self, state: State) -> Tuple[State, Dict]:
        pass

    def actor_critic_init(
        self, key: chex.PRNGKey
    ) -> Tuple[ActorCriticParams, ActingState]:
        num_devices = jax.local_device_count()

        (
            reset_key,
            actor_key,
            critic_key,
            outer_critic_key,
            acting_key,
        ) = jax.random.split(key, 5)
        reset_keys = jax.random.split(
            reset_key, num_devices * self.num_envs_per_device
        ).reshape((num_devices, self.num_envs_per_device, -1))
        env_state, timestep = jax.pmap(self.env.reset, axis_name="devices")(reset_keys)

        dummy_obs = jax.tree_util.tree_map(
            lambda x: x[None, ...], self.env.observation_spec().generate_value()
        )  # Add batch dim
        dummy_discount = jnp.zeros((), jnp.float32)
        params = ActorCriticParams(
            actor=self.actor_critic_networks.policy_network.init(
                actor_key, dummy_obs, dummy_discount
            ),
            critic=self.actor_critic_networks.value_network.init(
                critic_key, dummy_obs, dummy_discount
            ),
            outer_critic=None
            if self.actor_critic_networks.outer_value_network is None
            else self.actor_critic_networks.outer_value_network.init(
                outer_critic_key, dummy_obs, dummy_discount
            ),
        )
        acting_key_per_device = jax.random.split(acting_key, num_devices)
        acting_state = ActingState(
            env_state=env_state,
            timestep=timestep,
            acting_key=acting_key_per_device,
        )
        return params, acting_state

    def make_policy(
        self,
        policy_params: hk.Params,
        discount_factor: Optional[jnp.float_],
        deterministic: bool = False,
    ) -> Callable[[chex.Array, chex.PRNGKey], Tuple[chex.Array, Dict]]:
        policy_network = self.actor_critic_networks.policy_network
        parametric_action_distribution = (
            self.actor_critic_networks.parametric_action_distribution
        )

        def policy(
            observation: chex.Array, key: chex.PRNGKey
        ) -> Tuple[chex.Array, Dict]:
            logits = policy_network.apply(policy_params, observation, discount_factor)
            if deterministic:
                return parametric_action_distribution.mode(logits), {}
            raw_action = parametric_action_distribution.sample_no_postprocessing(
                logits, key
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_action)
            postprocessed_action = parametric_action_distribution.postprocess(
                raw_action
            )
            return postprocessed_action, {
                "log_prob": log_prob,
                "raw_action": raw_action,
            }

        return policy

    def rollout(
        self,
        actor_params: hk.Params,
        discount_factor: Optional[jnp.float_],
        acting_state: ActingState,
    ) -> Tuple[ActingState, Transition]:
        """Rollout for training purposes. Stop gradient at the end to prevent autograd from
        backpropagating through acting when doing meta-learning.
        Returns:
            shape (n_steps, batch_size_per_device, *)
        """
        policy = self.make_policy(
            policy_params=actor_params,
            discount_factor=discount_factor,
            deterministic=False,
        )

        def run_one_step(
            acting_state: ActingState, acting_key: chex.PRNGKey
        ) -> Tuple[ActingState, Transition]:
            timestep = acting_state.timestep
            action, policy_extras = policy(timestep.observation, acting_key)
            next_env_state, next_timestep = self.env.step(
                acting_state.env_state, action
            )
            next_acting_state = ActingState(
                env_state=next_env_state,
                timestep=next_timestep,
                acting_key=acting_key,
            )
            # No truncation in Snake
            truncation = jnp.zeros_like(next_timestep.discount)
            transition = Transition(
                observation=timestep.observation,
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                truncation=truncation,
                next_observation=next_timestep.observation,
                extras={
                    "policy_extras": policy_extras,
                },
            )
            return next_acting_state, transition

        def run_n_steps(
            acting_state: ActingState, _: None
        ) -> Tuple[ActingState, Transition]:
            acting_keys = jax.random.split(acting_state.acting_key, self.n_steps)
            acting_state, data = jax.lax.scan(run_one_step, acting_state, acting_keys)
            return acting_state, data

        assert self.batch_size_per_device % self.num_envs_per_device == 0
        num_sequence_rollouts = self.batch_size_per_device // self.num_envs_per_device
        acting_state, data = jax.lax.scan(
            run_n_steps, acting_state, None, num_sequence_rollouts
        )
        # Put time dimension (n_steps) first
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        # Merge mini-batches into batch_size_per_device
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x, (self.n_steps, self.batch_size_per_device, *x.shape[3:])
            ),
            data,
        )
        return acting_state, data
