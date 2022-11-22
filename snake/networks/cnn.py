from typing import Optional, Sequence

import chex
import haiku as hk
import jax
from jax import numpy as jnp

from snake.networks.actor_critic import ActorCriticNetworks, FeedForwardNetwork
from snake.networks.distribution import CategoricalParametricDistribution


def make_actor_critic_networks_cnn(
    num_actions: int,
    num_channels: int,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
    outer_critic: bool,
    embedding_size_actor: Optional[int],
    embedding_size_critic: Optional[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for Snake."""

    parametric_action_distribution = CategoricalParametricDistribution(
        num_actions=num_actions
    )
    policy_network = make_network_cnn(
        num_outputs=num_actions,
        mlp_units=policy_layers,
        conv_n_channels=num_channels,
        embedding_size=embedding_size_actor,
    )
    value_network = make_network_cnn(
        num_outputs=1,
        mlp_units=value_layers,
        conv_n_channels=num_channels,
        embedding_size=embedding_size_critic,
    )
    if outer_critic:
        outer_value_network: Optional[FeedForwardNetwork] = make_network_cnn(
            num_outputs=1,
            mlp_units=value_layers,
            conv_n_channels=num_channels,
            embedding_size=embedding_size_critic,
        )
    else:
        outer_value_network = None
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        outer_value_network=outer_value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_cnn(
    num_outputs: int,
    mlp_units: Sequence[int],
    conv_n_channels: int,
    embedding_size: Optional[int],
) -> FeedForwardNetwork:
    def network_fn(
        observation: chex.Array,
        discount_factor: Optional[jnp.float_],
    ) -> chex.Array:
        torso = hk.Sequential(
            [
                hk.Conv2D(conv_n_channels, (2, 2), 2),
                jax.nn.relu,
                hk.Conv2D(conv_n_channels, (2, 2), 1),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )
        if observation.ndim == 5:
            torso = jax.vmap(torso)
        x = torso(observation)
        if embedding_size:
            assert discount_factor is not None
            discount_factor = jnp.broadcast_to(discount_factor, (*x.shape[:-1], 1))
            embedding = hk.Linear(embedding_size)(discount_factor)
            x = jnp.concatenate([x, embedding], axis=-1)
        head = hk.nets.MLP((*mlp_units, num_outputs), activate_final=False)
        if num_outputs == 1:
            return jnp.squeeze(head(x), axis=-1)
        else:
            return head(x)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
