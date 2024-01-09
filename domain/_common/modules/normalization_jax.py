from typing import Tuple

import equinox as eqx
from equinox import nn
import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import Array

from einops import einsum


class DishTS(eqx.Module):
    reduce_mlayer: Array
    gamma: Array
    beta: Array

    def __init__(self, n_channels: int, seq_len: int, *, key: jrandom.PRNGKey):
        super().__init__()
        self.reduce_mlayer = jrandom.normal(
            key, (n_channels, seq_len, 2))/seq_len
        self.gamma = jnp.ones((n_channels, 1))
        self.beta = jnp.zeros((n_channels, 1))

    def __call__(self, x: Array):
        phil, phih, xil, xih = self.preget(x)
        x = self.forward_process(x, phil, xil)
        return x, (phih, xih)

    def denorm(self, x: Array, state: Tuple[Array, Array]):
        phih, xih = state
        x = ((x - self.beta) / self.gamma) * jnp.sqrt(xih + 1e-8) + phih
        return x

    def preget(self, x: Array):
        theta = einsum(x, self.reduce_mlayer, "... c l, c l two -> ... c two")
        theta = jnn.gelu(theta)
        phil = jax.lax.index_in_dim(theta, 0, axis=-1, keepdims=True)
        phih = jax.lax.index_in_dim(theta, 1, axis=-1, keepdims=True)

        n_reduced = x.shape[-1] - 1
        xil = jnp.sum((x - phil) ** 2, axis=-1, keepdims=True) / n_reduced
        xih = jnp.sum((x - phih) ** 2, axis=-1, keepdims=True) / n_reduced
        return phil, phih, xil, xih

    def forward_process(self, x: Array, phil: Array, xil: Array):
        temp = (x - phil)/jnp.sqrt(xil + 1e-8)
        rst = temp * self.gamma + self.beta
        return rst
