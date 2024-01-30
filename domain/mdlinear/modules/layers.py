from typing import Optional

import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float
import jax
from jax import nn as jnn
from jax.random import KeyArray
from jax import random as jrandom

from domain._common.modules.linear_jax import Linear
from domain._common.modules.attention_jax import MultiheadAttention


class TransformerBlock(eqx.Module):
    layer_norm1: nn.LayerNorm
    attn: MultiheadAttention
    layer_norm2: nn.LayerNorm
    mlp1: Linear
    mlp2: Linear

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        mlp_hidden: Optional[int] = None,
        *,
        key: KeyArray
    ):
        super(TransformerBlock, self).__init__()
        (attn_k, mlp1_k, mlp2_k) = jrandom.split(key, 3)

        mlp_hidden = mlp_hidden or 4 * d_model

        self.layer_norm1 = nn.LayerNorm((d_model,))
        self.attn = MultiheadAttention(n_heads, d_model, key=attn_k)
        self.layer_norm2 = nn.LayerNorm((d_model,))
        self.mlp1 = Linear(d_model, mlp_hidden, key=mlp1_k)
        self.mlp2 = Linear(mlp_hidden, d_model, key=mlp2_k)

    def __call__(self, x: Float[Array, "seq_len n_channels"]) -> Float[Array, "pred_len n_channels"]:
        q = jax.vmap(self.layer_norm1)(x)
        attn_x = self.attn(q, q, q) + x

        x = self.mlp1(jax.vmap(self.layer_norm2)(attn_x))
        x = jnn.silu(x)
        x = self.mlp2(x) + attn_x

        return x
