from functools import partial
from typing import Tuple
import math

import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float
import jax
from jax import numpy as jnp
from jax import nn as jnn
from jax.random import KeyArray
from jax import random as jrandom
from einops import rearrange

from domain.mamba.config import MambaConfig
from domain.mamba.modules.layers import MambaBlock


class Mamba(eqx.Module):
    blocks: MambaBlock
    layer_norm: nn.LayerNorm
    embedding: nn.Linear
    step_lin: nn.Linear
    combine_lin: nn.Linear
    pred_lin: nn.Linear

    d_inner: eqx.field(static=True)
    d_state: eqx.field(static=True)
    seq_len: eqx.field(static=True)
    label_len: eqx.field(static=True)
    pred_len: eqx.field(static=True)
    n_pred_steps: eqx.field(static=True)
    n_patches: eqx.field(static=True)
    patch_size: eqx.field(static=True)

    def __init__(self, config: MambaConfig.ModelConfig, *, key: KeyArray):
        super(Mamba, self).__init__()

        emb_k, step_lin_k, blocks_k, combine_lin_k, pred_lin_k = jrandom.split(
            key, 5
        )

        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.patch_size = config.patch_size
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.n_patches = (
            self.seq_len - self.patch_size) // self.patch_size + 1
        self.n_pred_steps = math.ceil(self.pred_len / self.patch_size) - 1
        self.combine_lin = nn.Linear(
            2 * config.d_model, config.d_model, key=combine_lin_k)
        self.pred_lin = nn.Linear(
            config.d_model * self.n_patches, self.pred_len, key=pred_lin_k)

        self.embedding = nn.Linear(
            self.patch_size, config.d_model, key=emb_k
        )
        blocks_ks = jrandom.split(blocks_k, config.n_blocks)
        self.blocks = eqx.filter_vmap(
            lambda k: self.make_component(config, key=k)
        )(blocks_ks)
        self.layer_norm = eqx.filter_vmap(
            lambda _: nn.LayerNorm((self.n_patches, config.d_model))
        )(jnp.arange(0, config.n_blocks))
        self.step_lin = nn.Linear(
            config.d_model * self.n_patches,
            self.seq_len,
            key=step_lin_k
        )

    def make_component(self, config: MambaConfig.ModelConfig,  *, key: jrandom.PRNGKey):
        return MambaBlock(
            d_model=config.d_model,
            d_state=self.d_state,
            d_conv=config.d_conv,
            d_inner=self.d_inner,
            d_dt=config.d_dt,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            key=key
        )

    def _slice_in_dim(self, data: Array, idx: int, dim: int) -> Array:
        return jax.lax.dynamic_slice_in_dim(data, idx * self.patch_size, self.patch_size, dim)

    def patch(self, x: Float[Array, "seq_len"], dim: int = -1) -> Float[Array, "patch_size"]:
        return jax.vmap(partial(self._slice_in_dim, data=x, dim=dim))(idx=jnp.arange(0, self.n_patches))

    def process_patches(
        self,
        x: Float[Array, "n_channels n_patches d_model"],
        ssm_state: Float[Array, "n_channels d_inner d_state"]
    ) -> Float[Array, "seq_len n_channels"]:
        dynamic_components, static_components = eqx.partition(
            self.blocks, eqx.is_array
        )
        dynamic_norm, _static_norm = eqx.partition(
            self.layer_norm, eqx.is_array
        )

        def layer_step(_carry, _dynamic) -> Tuple[
            Float[Array, "seq_len n_channels"],
            Float[Array, "seq_len n_channels"],
        ]:
            _hidden, _ssm_state = _carry
            _dynamic_component, _dynamic_norm = _dynamic
            component = eqx.combine(_dynamic_component, static_components)
            layer_norm = eqx.combine(_dynamic_norm, _static_norm)
            _upd_hidden, _upd_ssm_state = jax.vmap(component)(
                _hidden, _ssm_state
            )
            _upd_hidden = _hidden + _upd_hidden
            _upd_hidden = jax.vmap(layer_norm)(_upd_hidden)
            return (_upd_hidden, _upd_ssm_state), None

        step_init = (x, ssm_state)
        (hidden, ssm_state), _ = jax.lax.scan(
            layer_step, step_init, (dynamic_components, dynamic_norm))

        return hidden, ssm_state

    def backbone(self, x: Array):
        _, n_channels = x.shape

        # (n_patches, patch_size, n_total_channels)
        x = self.patch(x, dim=-2)
        x = rearrange(
            x, 'n_patches patch_size n_channels -> (n_patches n_channels) patch_size'
        )
        # (n_patches, n_total_channels, d_model)
        x = jax.vmap(self.embedding)(x)
        x = rearrange(
            x, '(n_patches n_channels) d_model -> n_channels n_patches d_model',
            n_patches=self.n_patches, n_channels=n_channels
        )

        ssm_state = jnp.zeros((n_channels, self.d_inner, self.d_state))
        hidden, ssm_state = self.process_patches(x, ssm_state)

        return hidden, ssm_state

    def __call__(
        self,
        x: Float[Array, "seq_len n_channels"],
    ) -> Float[Array, "seq_len n_channels"]:
        hidden, ssm_state = self.backbone(x)

        hidden = rearrange(
            hidden, 'n_channels n_patches d_model -> n_channels (n_patches d_model)'
        )
        preds = jax.vmap(self.step_lin)(hidden)
        preds = preds.transpose(-1, -2)

        return preds

    def predict(self, x: Float[Array, "seq_len n_channels"]) -> Float[Array, "pred_len n_channels"]:
        hidden, ssm_state = self.backbone(x)

        def pred_step(_carry, _) -> Tuple[
            Float[Array, "n_channels n_patches d_model"],
            Float[Array, "n_channels d_inner d_state"],
        ]:
            _hidden, _ssm_state, _pred_hidden = _carry
            _upd_hidden, _upd_ssm_state = self.process_patches(
                _hidden, _ssm_state)

            _pred_hidden = jnp.concatenate(
                (_pred_hidden, _upd_hidden), axis=-1)
            _n_channels, _n_patches = _pred_hidden.shape[:2]
            _pred_hidden = rearrange(
                _pred_hidden, "n_channels n_patches d_model -> (n_channels n_patches) d_model", n_channels=_n_channels, n_patches=_n_patches)
            _pred_hidden = jax.vmap(self.combine_lin)(_pred_hidden)
            _pred_hidden = rearrange(
                _pred_hidden, "(n_channels n_patches) d_model -> n_channels n_patches d_model", n_channels=_n_channels, n_patches=_n_patches)

            return (_upd_hidden, _upd_ssm_state, _pred_hidden), None

        carry, _ = jax.lax.scan(
            pred_step, (hidden, ssm_state, jnp.copy(hidden)), None, length=self.n_pred_steps
        )
        _, res_ssm_state, res_hidden = carry

        res_hidden, res_ssm_state = self.process_patches(res_hidden, res_ssm_state)
        res_hidden = rearrange(
            res_hidden, 'n_channels n_patches d_model -> n_channels (n_patches d_model)'
        )
        res_hidden = jax.vmap(self.pred_lin)(res_hidden)
        res_hidden = res_hidden.transpose(-1, -2)

        return res_hidden
