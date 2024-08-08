import equinox as eqx
from equinox import nn
from jaxtyping import Float, Array
import jax
from jax import (
    numpy as jnp,
    nn as jnn,
    random as jrandom
)
from einops import rearrange, repeat

from domain._common.modules.linear_jax import Linear
from domain._common.modules.attention_jax import MultiheadAttention
import math

from typing import Optional


class SSMBlock(eqx.Module):
    in_lin: Linear
    conv: nn.Conv1d
    out_lin: Linear
    param_lin: Linear
    dt_lin: Linear
    A_log: Array
    D: Array

    d_dt: int = eqx.field(static=True)
    d_conv: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        d_inner: int,
        d_dt: int,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
        *,
        key: jrandom.KeyArray
    ):
        super(SSMBlock, self).__init__()
        in_lin_k, conv_k, params_k, out_k = jrandom.split(key, 4)

        dt_min = dt_min or 0.001
        dt_max = dt_max or 0.1

        self.d_conv = d_conv
        self.init_params(d_inner, d_state, d_dt, dt_min, dt_max, key=params_k)

        self.in_lin = Linear(d_model, 2 * d_inner, key=in_lin_k)
        self.conv = nn.Conv1d(d_inner, d_inner, self.d_conv, key=conv_k)
        self.out_lin = Linear(d_inner, d_model, key=out_k)

    def init_params(
        self,
        d_inner: int,
        d_state: int,
        d_dt: int,
        dt_min: float,
        dt_max: float,
        *,
        key: jrandom.KeyArray
    ) -> Float[Array, "d_inner"]:
        param_k, dt_lin_k, dt_weight_k, dt_bias_k = jrandom.split(key, 4)

        self.d_dt = d_dt
        self.d_state = d_state

        self.param_lin = Linear(
            d_inner, self.d_dt + 2 * d_state, key=param_k
        )
        self.D = jnp.ones((d_inner,))

        dt_lin = Linear(self.d_dt, d_inner, key=dt_lin_k)
        dt_init_std = self.d_dt ** -0.5
        dt_lin = eqx.tree_at(lambda l: l.weight, dt_lin, jrandom.uniform(
            dt_weight_k,
            dt_lin.weight.shape,
            minval=-dt_init_std,
            maxval=dt_init_std
        ))
        dt_bias = jnp.exp(
            jrandom.uniform(dt_bias_k, dt_lin.bias.shape) *
            (math.log(dt_max) - math.log(dt_min))
        ).clip(min=dt_min)
        inv_dt_bias = dt_bias + jnp.log(-jnp.expm1(-dt_bias))
        dt_lin = dt_lin = eqx.tree_at(lambda l: l.bias, dt_lin, inv_dt_bias)
        self.dt_lin = dt_lin

        A = repeat(
            jnp.arange(1, d_state + 1, dtype=jnp.float32),
            "n -> d n",
            d=d_inner
        )
        self.A_log = jnp.log(A)

    def ssm(self, ssm_state, data):
        x, z, params = data
        dA, dB, C = params

        ssm_state = ssm_state * dA + jnp.expand_dims(x, -1) * dB
        y = jnp.einsum("cdn,cn->cd", ssm_state, C)
        y = y + self.D * x
        y = y * jnn.gelu(z)

        return ssm_state, y

    def pad_for_conv(self, x: Array, axis: int = -1):
        padding = [(0, 0) for _ in range(x.ndim)]
        padding[axis] = (self.d_conv // 2, math.ceil(self.d_conv / 2) - 1)

        return jnp.pad(x, padding, mode='edge')

    def __call__(
        self,
        x: Float[Array, "n_channels n_patches d_model"],
        ssm_state: Float[Array, "n_channels d_inner"] = None,
    ) -> Float[Array, "n_channels n_patches d_model"]:
        xz = self.in_lin(x)
        x, z = jnp.split(xz, 2, axis=-1)

        x = self.pad_for_conv(x, -2)
        x = x.swapaxes(-1, -2)
        x = jax.vmap(self.conv)(x)
        x = x.swapaxes(-1, -2)
        x = jnn.gelu(x)

        x_db = self.param_lin(x)
        params = jnp.split(
            x_db,
            (self.d_dt, self.d_dt + self.d_state),
            axis=-1
        )
        dt, B, C = params
        dt = self.dt_lin(dt)  # (d_inner)
        dt = jnn.softplus(dt)
        A = -jnp.exp(self.A_log)  # (d_inner, d_state)

        dA = jnp.exp(jnp.einsum("cbd,dn->bcdn", dt, A))
        dB = jnp.einsum("cbd,cbn->bcdn", dt, B)

        z = jnn.gelu(z)
        C = C.swapaxes(-2, -3)
        x = x.swapaxes(-2, -3)
        z = z.swapaxes(-2, -3)

        params = (dA, dB, C)
        data = (x, z, params)
        ssm_state, y = jax.lax.scan(self.ssm, ssm_state, data)

        y = self.out_lin(y)
        y = y.swapaxes(-2, -3)

        return y, ssm_state


class Mixer(eqx.Module):
    attn: MultiheadAttention
    ln1: nn.LayerNorm
    mlp1: Linear
    mlp2: Linear
    ln2: nn.LayerNorm

    def __init__(
        self,
        seq_len: int,
        n_heads: int,
        d_model: int,
        *,
        key: jrandom.KeyArray
    ):
        super(Mixer, self).__init__()
        attn_k, mlp1_k, mlp2_k = jrandom.split(key, 3)
        self.ln1 = nn.LayerNorm((seq_len, d_model,))
        self.attn = MultiheadAttention(n_heads, d_model, key=attn_k)

        self.ln2 = nn.LayerNorm((seq_len, d_model,))
        self.mlp1 = Linear(d_model, 2*d_model, key=mlp1_k)
        self.mlp2 = Linear(2*d_model, d_model, key=mlp2_k)

    def __call__(self, x: Float[Array, "n_channels n_patches d_model"]) -> Float[Array, "n_channels n_patches d_model"]:
        x = x + jax.vmap(self.ln1)(self.attn(x, x, x))
        x = x + jax.vmap(self.ln2)(self.mlp2(jnn.silu(self.mlp1(x))))

        return x
