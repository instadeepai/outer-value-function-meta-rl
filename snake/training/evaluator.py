import functools
from typing import Callable, Dict, Optional, Tuple

import chex
import haiku as hk
import jax
import jumanji.types
from jax import numpy as jnp

from snake.agent import ActorCriticAgent
from snake.training.types import TrainingState


class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: jumanji.Environment,
        actor_critic_agent: ActorCriticAgent,
        total_num_eval: int,
        key: chex.PRNGKey,
        deterministic: bool,
    ):
        self._eval_env = eval_env
        self._actor_critic_agent = actor_critic_agent

        num_devices = jax.local_device_count()
        self._num_devices = num_devices
        assert total_num_eval % num_devices == 0
        self._total_num_eval = total_num_eval
        self._num_eval_per_device = total_num_eval // num_devices

        self._key = key
        self._deterministic = deterministic
        self.generate_eval_unroll = jax.pmap(
            self._generate_eval_unroll, axis_name="devices"
        )

    def _unroll_one_step(
        self,
        policy: Callable[[chex.Array, chex.PRNGKey], Tuple[chex.Array, Dict]],
        carry: Tuple[jumanji.env.State, jumanji.types.TimeStep],
        key: chex.PRNGKey,
    ) -> Tuple[Tuple[jumanji.env.State, jumanji.types.TimeStep], None]:
        state, timestep = carry
        observation = jax.tree_util.tree_map(
            lambda x: x[None, ...], timestep.observation
        )
        action, _ = policy(observation, key)
        next_state, next_timestep = self._eval_env.step(state, jnp.squeeze(action))
        return (next_state, next_timestep), None

    def _generate_eval_one_episode(
        self,
        policy_params: hk.Params,
        discount_factor: Optional[jnp.float_],
        key: chex.PRNGKey,
    ) -> Tuple[Dict, Dict]:
        stochastic_policy = self._actor_critic_agent.make_policy(
            policy_params,
            discount_factor,
            deterministic=False,
        )
        determinist_policy = self._actor_critic_agent.make_policy(
            policy_params,
            discount_factor,
            deterministic=True,
        )

        def cond_fun(
            carry: Tuple[
                jumanji.env.State,
                jumanji.types.TimeStep,
                chex.PRNGKey,
                jnp.float32,
                jnp.int32,
            ]
        ) -> jnp.bool_:
            _, timestep, *_ = carry
            return ~timestep.last()

        def body_fun(
            policy: Callable[[chex.Array, chex.PRNGKey], Tuple[chex.Array, Dict]],
            carry: Tuple[
                jumanji.env.State,
                jumanji.types.TimeStep,
                chex.PRNGKey,
                jnp.float32,
                jnp.int32,
            ],
        ) -> Tuple[
            jumanji.env.State,
            jumanji.types.TimeStep,
            chex.PRNGKey,
            jnp.float32,
            jnp.int32,
        ]:
            state, timestep, key, return_, count = carry
            key, step_key = jax.random.split(key)
            (state, timestep), _ = self._unroll_one_step(
                policy, (state, timestep), step_key
            )
            return_ += timestep.reward
            count += 1
            return state, timestep, key, return_, count

        (
            reset_key_stochastic,
            reset_key_determinist,
            init_key_stochastic,
            init_key_determinist,
        ) = jax.random.split(key, 4)
        state_stochastic, timestep_stochastic = self._eval_env.reset(
            reset_key_stochastic
        )
        _, _, _, return_stochastic, count_stochastic = jax.lax.while_loop(
            cond_fun,
            functools.partial(body_fun, stochastic_policy),
            (
                state_stochastic,
                timestep_stochastic,
                init_key_stochastic,
                jnp.float32(0),
                jnp.int32(0),
            ),
        )
        eval_metrics_stochastic = {
            "episode_reward": return_stochastic,
            "episode_length": count_stochastic,
        }
        state_determinist, timestep_determinist = self._eval_env.reset(
            reset_key_stochastic
        )
        _, _, _, return_deterministic, count_deterministic = jax.lax.while_loop(
            cond_fun,
            functools.partial(body_fun, determinist_policy),
            (
                state_determinist,
                timestep_determinist,
                init_key_determinist,
                jnp.float32(0),
                jnp.int32(0),
            ),
        )
        eval_metrics_deterministic = {
            "episode_reward": return_deterministic,
            "episode_length": count_deterministic,
        }
        return eval_metrics_stochastic, eval_metrics_deterministic

    def _generate_eval_unroll(
        self,
        policy_params: hk.Params,
        discount_factor: Optional[jnp.float_],
        key: chex.PRNGKey,
    ) -> Dict:

        keys = jax.random.split(key, self._num_eval_per_device)
        eval_metrics = jax.vmap(
            self._generate_eval_one_episode, in_axes=(None, None, 0)
        )(
            policy_params,
            discount_factor,
            keys,
        )
        eval_metrics: Dict = jax.lax.pmean(
            jax.tree_util.tree_map(jnp.mean, eval_metrics),
            axis_name="devices",
        )

        return eval_metrics

    def run_evaluation(self, training_state: TrainingState) -> Dict:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        unroll_keys = jax.random.split(unroll_key, self._num_devices)
        if training_state.meta_params is not None:
            discount_factor = jax.nn.sigmoid(training_state.meta_params.gamma)
        else:
            discount_factor = None
        eval_metrics_stochastic, eval_metrics_determinist = self.generate_eval_unroll(
            training_state.params.actor,
            discount_factor,
            unroll_keys,
        )
        stochastic_metrics = eval_metrics_stochastic
        determinist_metrics = eval_metrics_determinist
        metrics = {
            **{
                key + "_stochastic_policy": value
                for key, value in stochastic_metrics.items()
            },
            **{
                key + "_determinist_policy": value
                for key, value in determinist_metrics.items()
            },
        }
        return metrics
