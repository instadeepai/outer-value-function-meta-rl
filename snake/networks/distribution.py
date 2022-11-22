"""Copied from Brax and adapted with typing."""
import abc
from typing import Any

import chex
import jax
from jax import numpy as jnp


class Postprocessor(abc.ABC):
    def forward(self, x: chex.Array) -> chex.Array:
        raise NotImplementedError

    def inverse(self, y: chex.Array) -> chex.Array:
        raise NotImplementedError

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        raise NotImplementedError


class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(
        self,
        param_size: int,
        postprocessor: Postprocessor,
        event_ndims: int,
        reparametrizable: bool,
    ):
        """Abstract class for parametric (action) distribution.
        Specifies how to transform distribution parameters (i.e. actor output)
        into a distribution over actions.
        Args:
            param_size: size of the parameters for the distribution
            postprocessor: bijector which is applied after sampling (in practice, it's
                tanh or identity)
            event_ndims: rank of the distribution sample (i.e. action)
            reparametrizable: is the distribution reparametrizable
        """
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims  # rank of events
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters: chex.Array) -> Any:
        """Creates distribution from parameters."""

    @property
    def param_size(self) -> int:
        return self._param_size

    @property
    def reparametrizable(self) -> bool:
        return self._reparametrizable

    def postprocess(self, event: chex.Array) -> chex.Array:
        return self._postprocessor.forward(event)

    def inverse_postprocess(self, event: chex.Array) -> chex.Array:
        return self._postprocessor.inverse(event)

    def sample_no_postprocessing(
        self, parameters: chex.Array, seed: chex.PRNGKey
    ) -> Any:
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters: chex.Array, seed: chex.PRNGKey) -> chex.Array:
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters: chex.Array) -> chex.Array:
        """Returns the mode of the postprocessed distribution."""
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters: chex.Array, raw_actions: chex.Array) -> chex.Array:
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(raw_actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(raw_actions)
        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension
        return log_probs

    def entropy(self, parameters: chex.Array, seed: chex.PRNGKey) -> chex.Array:
        """Return the entropy of the given distribution."""
        dist = self.create_dist(parameters)
        entropy = dist.entropy()
        entropy += self._postprocessor.forward_log_det_jacobian(dist.sample(seed=seed))
        if self._event_ndims == 1:
            entropy = jnp.sum(entropy, axis=-1)
        return entropy

    def kl_divergence(
        self, parameters: chex.Array, other_parameters: chex.Array
    ) -> chex.Array:
        """KL divergence is invariant with respect to transformation by the same bijector."""
        dist = self.create_dist(parameters)
        other_dist = self.create_dist(other_parameters)
        return dist.kl_divergence(other_dist)


class IdentityBijector(Postprocessor):
    """Identity Bijector."""

    def forward(self, x: chex.Array) -> chex.Array:
        return x

    def inverse(self, y: chex.Array) -> chex.Array:
        return y

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        return jnp.zeros_like(x, x.dtype)


class CategoricalDistribution:
    """Categorical distribution."""

    def __init__(self, logits: chex.Array):
        self.logits = logits
        self.num_actions = jnp.shape(logits)[-1]

    def sample(self, seed: chex.PRNGKey) -> chex.Array:
        return jax.random.categorical(seed, self.logits)

    def mode(self) -> chex.Array:
        return jnp.argmax(self.logits, axis=-1)

    def log_prob(self, x: chex.Array) -> chex.Array:
        value_one_hot = jax.nn.one_hot(x, self.num_actions)
        mask_outside_domain = jnp.logical_or(x < 0, x > self.num_actions - 1)
        safe_log_probs = jnp.where(
            value_one_hot == 0,
            jnp.zeros((), dtype=self.logits.dtype),
            jax.nn.log_softmax(self.logits) * value_one_hot,
        )
        return jnp.where(
            mask_outside_domain,
            -jnp.inf,
            jnp.sum(safe_log_probs, axis=-1),
        )

    def entropy(self) -> chex.Array:
        log_probs = jax.nn.log_softmax(self.logits)
        probs = jnp.exp(log_probs)
        return -jnp.sum(jnp.where(probs == 0, 0.0, probs * log_probs), axis=-1)

    def kl_divergence(self, other: "CategoricalDistribution") -> chex.Array:
        log_probs = jax.nn.log_softmax(self.logits)
        probs = jnp.exp(log_probs)
        log_probs_other = jax.nn.log_softmax(other.logits)
        return jnp.sum(
            jnp.where(probs == 0, 0.0, probs * (log_probs - log_probs_other)), axis=-1
        )


class CategoricalParametricDistribution(ParametricDistribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, num_actions: int):
        """Initialize the distribution.
        Args:
            num_actions: the number of actions.
        """
        postprocessor: Postprocessor
        postprocessor = IdentityBijector()
        super().__init__(
            param_size=num_actions,
            postprocessor=postprocessor,
            event_ndims=0,
            reparametrizable=True,
        )

    def create_dist(self, parameters: chex.Array) -> CategoricalDistribution:
        return CategoricalDistribution(logits=parameters)
