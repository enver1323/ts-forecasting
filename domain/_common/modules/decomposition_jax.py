import math
from typing import Tuple

import numpy as np
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import Array

import equinox as eqx
from equinox import nn

from domain._common.modules.linear_jax import Linear


class DeTrend(eqx.Module):
    linear: eqx.Module

    def __init__(self, in_features: int, *, key: jrandom.PRNGKey):
        super(DeTrend, self).__init__()

        self.linear = Linear(in_features, in_features, key=key)

    def __call__(self, data: Array) -> Array:
        trend = jnn.softmax(self.linear(data))
        trend = trend * data
        return trend


class EMA(eqx.Module):
    alpha: float = eqx.field(static=True)

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def __call__(self, data: Array, axis: int = -1) -> Array:
        powers = np.arange(data.shape[axis])
        powers = powers[::-1]
        powers = np.power(1 - self.alpha, powers)
        initial_powers = powers.copy()
        powers[1:] = powers[1:] * self.alpha
        s = powers * data
        return jnp.cumsum(s, axis=0) / initial_powers


class MovingAverage(eqx.Module):
    avg_pool: eqx.Module
    kernel_size: int = eqx.field(static=True)

    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=stride, padding=0
        )

    def __call__(self, data: Array, axis: int = -1) -> Array:
        pad_shape = [(0, 0) for _ in range(data.ndim)]
        pad_shape[axis] = (
            self.kernel_size // 2, math.ceil(self.kernel_size / 2) - 1
        )
        data = jnp.pad(data, pad_shape, mode='edge')
        x = self.avg_pool(data.swapaxes(-1, axis))
        x = x.swapaxes(-1, axis)

        return x


class SeriesDecomposition(eqx.Module):
    decomposer: eqx.Module

    def __init__(self, decomposer: eqx.Module):
        super(SeriesDecomposition, self).__init__()

        self.decomposer = decomposer

    def __call__(self, data: Array) -> Tuple[Array, Array]:
        trend = self.decomposer(data)
        seasonality = data - trend
        return trend, seasonality
