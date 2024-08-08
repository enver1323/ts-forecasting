import equinox as eqx
from equinox import nn, static_field
from jaxtyping import Array, Float
import jax
from jax import numpy as jnp, nn as jnn
from jax.random import KeyArray
from jax import random as jrandom

from domain.ilinear.config import ILinearConfig


class ILinear(eqx.Module):
    seq_hid: nn.Linear
    seq_pred: nn.Linear

    ch_hid: nn.Linear
    ch_pred: nn.Linear

    pred_len: int = static_field()

    def __init__(self, config: ILinearConfig.ModelConfig, *, key: KeyArray):
        super(ILinear, self).__init__()

        shk, spk, chk, cpk = jrandom.split(key, 4)

        self.pred_len = config.pred_len

        self.seq_hid = nn.Linear(config.seq_len, config.hidden_size, key=shk)
        self.seq_pred = nn.Linear(config.hidden_size, config.pred_len, key=spk)

        self.ch_hid = nn.Linear(config.n_channels, config.hidden_size, key=chk)
        self.ch_pred = nn.Linear(config.hidden_size, config.n_channels, key=cpk)

    def __call__(self, x: Float[Array, "context_size n_channels"]) -> Float[Array, "context_size n_channels"]:
        x = jax.vmap(self.ch_hid)(x)
        x = jnn.elu(x)
        x = jax.vmap(self.ch_pred)(x)
        x = jnn.elu(x)

        x = x.transpose(-1, -2)

        x = jax.vmap(self.seq_hid)(x)
        x = jnn.elu(x)
        x = jax.vmap(self.seq_pred)(x)

        x = x.transpose(-1, -2)

        return x
