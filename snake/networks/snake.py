from typing import Optional, Sequence

from snake.networks import cnn
from snake.networks.actor_critic import ActorCriticNetworks


def make_actor_critic_networks_snake(
    num_channels: int,
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
    outer_critic: bool,
    embedding_size_actor: Optional[int],
    embedding_size_critic: Optional[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for Snake."""

    return cnn.make_actor_critic_networks_cnn(
        num_actions=4,
        num_channels=num_channels,
        policy_layers=policy_layers,
        value_layers=value_layers,
        outer_critic=outer_critic,
        embedding_size_actor=embedding_size_actor,
        embedding_size_critic=embedding_size_critic,
    )
