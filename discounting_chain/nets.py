from typing import Optional, Tuple

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

    def __init__(self, conv_strides) -> None:
        super().__init__(name="atari_torso")
        self._network = hk.Sequential(
            [
                hk.Conv2D(32, [2, 2], conv_strides[0]),
                jax.nn.relu,
                hk.Conv2D(32, [2, 2], conv_strides[1]),
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


def create_linear_forward_fn(
    environment_spec: EnvironmentSpec, double_value_head: bool = False
):
    # A minimal forward function, where the actor is a linear layer.
    # The critic for this function is not intended to be used.
    def forward_fn(
        obs: jnp.ndarray,
    ) -> Tuple[tfp.distributions.Distribution, chex.ArrayTree]:
        actor_network = jax_networks.CategoricalHead(
            num_values=environment_spec.actions.num_values,
            w_init=hk.initializers.VarianceScaling(1e-4),
        )
        policy = actor_network(obs)
        # value function is a dummy if using linear forward fn
        val = hk.Linear(1, w_init=hk.initializers.VarianceScaling(1e-4))(obs) * jnp.nan
        value = val, val if double_value_head else val
        return policy, value

    return hk.without_apply_rng(hk.transform(forward_fn))


def create_forward_fn(
    environment_spec: EnvironmentSpec,
    actor_hidden_layers: Tuple[int, ...] = (32, 32),
    critic_hidden_layers: Tuple[int, ...] = (64, 64),
    meta_critic_hidden_layers: Optional = None,
    torso_mlp_units=(32, 32),
    conv_strides=(2, 1),
    double_value_head: bool = False,
) -> hk.Transformed:
    assert len(environment_spec.observations.shape) in [1, 3]

    torso = (
        lambda: MiniAtariTorso(conv_strides)
        if len(environment_spec.observations.shape) == 3
        else hk.nets.MLP(torso_mlp_units)
    )

    def forward_fn(
        obs: jnp.ndarray,
    ) -> Tuple[tfp.distributions.Distribution, chex.ArrayTree]:
        enc_crit = torso()(obs)
        enc_crit_torso = enc_crit
        critic_network: hk.Module = hk.nets.MLP(
            critic_hidden_layers, activate_final=True
        )
        enc_crit = critic_network(enc_crit)
        value = hk.Linear(1, w_init=hk.initializers.VarianceScaling(1e-4))(enc_crit)

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
        value = jnp.squeeze(value, axis=-1)
        if double_value_head:
            if meta_critic_hidden_layers:
                critic_network_2 = hk.nets.MLP(
                    meta_critic_hidden_layers, activate_final=True
                )
                value_2 = critic_network_2(jax.lax.stop_gradient(enc_crit_torso))
            else:
                value_2 = jax.lax.stop_gradient(enc_crit)
            value_2 = hk.Linear(1, w_init=hk.initializers.VarianceScaling(1e-4))(
                value_2
            )
            value_2 = jnp.squeeze(value_2, axis=-1)
            value = value, value_2
        return policy, value

    return hk.without_apply_rng(hk.transform(forward_fn))
