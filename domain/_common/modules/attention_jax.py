import warnings
import math
from functools import partial
from typing import Optional, Union

import jax
from jax import (
    random as jrandom,
    numpy as jnp
)
from jax.random import PRNGKeyArray
from equinox import Module, field
from equinox.nn import Dropout
from jaxtyping import Array, Float, Bool
from einops import rearrange

from domain._common.modules.linear_jax import Linear

def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:
    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("...sd,...Sd->...sS", query, key)
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

    return jax.nn.softmax(logits, axis=-1)  # pyright: ignore

def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    *,
    key: Optional[PRNGKeyArray] = None,
) -> Float[Array, "q_seq v_size"]:
    weights = dot_product_attention_weights(query, key_, mask)
    attn = jnp.einsum("...sS,...Sd->...sd", weights, value)
    return attn


class MultiheadAttention(Module):
    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear

    num_heads: int = field(static=True)
    query_size: int = field(static=True)
    key_size: int = field(static=True)
    value_size: int = field(static=True)
    output_size: int = field(static=True)
    qk_size: int = field(static=True)
    vo_size: int = field(static=True)
    use_query_bias: bool = field(static=True)
    use_key_bias: bool = field(static=True)
    use_value_bias: bool = field(static=True)
    use_output_bias: bool = field(static=True)

    def __init__(
        self,
        num_heads: int,
        query_size: int,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        qkey, kkey, vkey, okey = jrandom.split(key, 4)

        if key_size is None:
            key_size = query_size
        if value_size is None:
            value_size = query_size
        if qk_size is None:
            qk_size = query_size // num_heads
        if vo_size is None:
            vo_size = query_size // num_heads
        if output_size is None:
            output_size = query_size

        self.query_proj = Linear(
            query_size, num_heads * qk_size, use_bias=use_query_bias, key=qkey
        )
        self.key_proj = Linear(
            key_size, num_heads * qk_size, use_bias=use_key_bias, key=kkey
        )
        self.value_proj = Linear(
            value_size, num_heads * vo_size, use_bias=use_value_bias, key=vkey
        )
        self.output_proj = Linear(
            num_heads * vo_size, output_size, use_bias=use_output_bias, key=okey
        )

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias

    @jax.named_scope("eqx.nn.MultiheadAttention")
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"]
        ] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "q_seq o_size"]:
        query_seq_length = query.shape[-2]
        k_seq_length = key_.shape[-2]
        v_seq_length = value.shape[-2]
        if k_seq_length != v_seq_length:
            # query length can be different
            raise ValueError(
                "key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        keys = None if key is None else jax.random.split(
            key, query_heads.shape[1])
        
        attn = dot_product_attention(
            query_heads, key_heads, value_heads, mask=mask, key=keys
        )
        attn = rearrange(attn, "... h d -> ... (h d)")

        return self.output_proj(attn)

    def _project(self, proj: Linear, x: Array):
        projection = proj(x)
        return rearrange(projection, "... (h d) -> ... h d", h=self.num_heads)
