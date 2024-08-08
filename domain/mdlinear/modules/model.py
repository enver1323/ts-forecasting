from functools import partial
import math

import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float
import jax
from jax import numpy as jnp
from jax import nn as jnn
from jax.random import KeyArray
from jax import random as jrandom
from statsmodels.tsa.seasonal import MSTL
from einops import rearrange

from domain._common.modules.linear_jax import Linear
from domain._common.modules.attention_jax import MultiheadAttention
from domain.mdlinear.config import MDLinearConfig
from domain._common.modules.decomposition_jax import SeriesDecomposition, MovingAverage, EMA
from domain._common.modules.normalization_jax import RevIN, DishTS
from domain.mdlinear.modules.layers import TransformerBlock


class MDLinear(eqx.Module):
    embed: Linear
    # attn: TransformerBlock
    # pred: Linear

    def __init__(self, config: MDLinearConfig.ModelConfig, *, key: KeyArray):
        super(MDLinear, self).__init__()
        (emb_k, attn_k, pred_k) = jrandom.split(key, 3)

        self.embed = Linear(config.seq_len + 2, config.pred_len + 2, key=emb_k)
        # self.embed = Linear(config.seq_len + 2, config.d_model, key=emb_k)
        # self.attn = TransformerBlock(
        #     config.n_heads, config.d_model, key=attn_k
        # )
        # self.pred = Linear(config.d_model, config.pred_len + 2, key=pred_k)

    def __call__(self, x: Float[Array, "seq_len n_channels"]) -> Float[Array, "pred_len n_channels"]:
        mean = jnp.mean(x, axis=-2)
        std = jnp.std(x, axis=-2)
        x = (x - mean) / std

        data = jnp.stack([mean, std], axis=-2)
        data = jnp.concatenate([data, x], axis=-2)

        pred = self.embed(data.swapaxes(-1, -2))
        # pred = self.attn(pred)
        # pred = jnn.gelu(pred)
        # pred = self.pred(pred)
        pred = pred.swapaxes(-1, -2)

        mean = pred[0]
        std = pred[1]
        pred = pred[2:]

        pred = pred * std + mean

        return pred, mean, std
