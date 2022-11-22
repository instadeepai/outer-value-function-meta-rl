from typing import Tuple

import chex
import haiku as hk
import jax
from acme.jax import networks as jax_networks
from acme.specs import EnvironmentSpec
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp


class MiniAtariTorso(jax_networks.base.Module):
    """A network in the style of `acme.jax.networks.AtariTorso` but for smaller image sizes,
    like with Snake from Jumanji."""

    def __init__(self) -> None:
        super().__init__(name="atari_torso")
        self._network = hk.Sequential(
            [
                hk.Conv2D(32, [2, 2], 2),
                jax.nn.relu,
                hk.Conv2D(32, [2, 2], 1),
                jax.nn.relu,
            ]
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        inputs_rank = jnp.ndim(inputs)
        batched_inputs = inputs_rank == 4
        if inputs_rank < 3 or inputs_rank > 4:
            raise ValueError("Expected input BHWC or HWC. Got rank %d" % inputs_rank)

        outputs = self._network(inputs)

        if batched_inputs:
            return jnp.reshape(outputs, [outputs.shape[0], -1])  # [B, D]
        return jnp.reshape(outputs, [-1])  # [D]


def create_forward_fn(
    environment_spec: EnvironmentSpec,
    actor_hidden_layers: Tuple[int, ...] = (64, 64),
    critic_hidden_layers: Tuple[int, ...] = (64, 64),
    torso_mlp_units=(32, 32),
) -> Tuple[hk.Transformed, hk.Transformed]:
    assert len(environment_spec.observations.shape) in [1, 3]

    torso = (
        lambda: MiniAtariTorso()
        if len(environment_spec.observations.shape) == 3
        else hk.nets.MLP(torso_mlp_units)
    )

    def policy_forward_fn(
        obs: jnp.ndarray,
    ) -> Tuple[tfp.distributions.Distribution, chex.ArrayTree]:
        enc_act = torso()(obs)
        actor_network: hk.Module = hk.Sequential(
            [
                hk.nets.MLP(actor_hidden_layers, activate_final=True),
                jax_networks.CategoricalHead(
                    num_values=environment_spec.actions.num_values
                ),
            ],
            name="actor",
        )
        policy = actor_network(enc_act)
        return policy

    def critic_forward_fn(
        obs: jnp.ndarray,
    ) -> Tuple[tfp.distributions.Distribution, chex.ArrayTree]:
        enc_crit = torso()(obs)
        critic_network: hk.Module = hk.nets.MLP(
            critic_hidden_layers, activate_final=True
        )
        enc_crit = critic_network(enc_crit)
        value = hk.Linear(1, w_init=hk.initializers.VarianceScaling(1e-4))(enc_crit)
        value = jnp.squeeze(value, axis=-1)
        return value

    return hk.without_apply_rng(hk.transform(policy_forward_fn)), hk.without_apply_rng(
        hk.transform(critic_forward_fn)
    )
