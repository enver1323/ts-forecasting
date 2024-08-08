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

from domain._common.modules.linear_jax import Linear
from domain._common.modules.normalization_jax import DishTS
from domain.ssm.config import SSMConfig
from domain.ssm.modules.layers import SSMBlock, Mixer


class SSM(eqx.Module):
    blocks: SSMBlock
    layer_norm1: nn.LayerNorm
    layer_norm2: nn.LayerNorm
    embedding: Linear
    step_lin: Linear
    mlp1: Linear
    mlp2: Linear
    norm: DishTS
    step_gru: nn.GRUCell
    dropout: nn.Dropout
    pos_emb: Array
    ch_emb: Array

    d_inner: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)
    seq_len: int = eqx.field(static=True)
    label_len: int = eqx.field(static=True)
    pred_len: int = eqx.field(static=True)
    n_pred_steps: int = eqx.field(static=True)
    n_patches: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    def __init__(self, config: SSMConfig.ModelConfig, *, key: KeyArray):
        super(SSM, self).__init__()
        (
            emb_k,
            pred_k,
            blocks_k,
            norm_k,
            mlp1_k,
            mlp2_k,
        ) = jrandom.split(key, 6)

        self.seq_len = config.seq_len
        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.patch_size = config.patch_size
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.n_patches = (
            self.seq_len - self.patch_size) // self.patch_size + 1
        self.n_pred_steps = math.ceil(self.pred_len / self.patch_size) - 1

        self.embedding = Linear(
            self.patch_size, config.d_model, key=emb_k
        )
        blocks_ks = jrandom.split(blocks_k, config.n_blocks)
        self.blocks = eqx.filter_vmap(
            lambda k: self.make_component(config, key=k)
        )(blocks_ks)
        self.layer_norm1 = eqx.filter_vmap(
            lambda _: nn.LayerNorm((self.n_patches, config.d_model))
        )(jnp.arange(0, config.n_blocks))
        self.layer_norm2 = eqx.filter_vmap(
            lambda _: nn.LayerNorm((self.n_patches, config.d_model))
        )(jnp.arange(0, config.n_blocks))
        # self.step_lin = Linear(
        #     config.d_model, self.patch_size, key=pred_k
        # )
        norm_ks = jrandom.split(norm_k, config.n_blocks)
        self.norm = jax.vmap(lambda k: DishTS(
            self.n_patches, config.d_model, key=k))(norm_ks)

        mlp1_ks = jrandom.split(mlp1_k, config.n_blocks)
        self.mlp1 = jax.vmap(lambda k: Linear(
            config.d_model, config.d_model * 4, key=k))(mlp1_ks)
        mlp2_ks = jrandom.split(mlp2_k, config.n_blocks)
        self.mlp2 = jax.vmap(lambda k: Linear(
            4 * config.d_model, config.d_model, key=k))(mlp2_ks)

        self.n_pred_steps = math.ceil(self.pred_len / self.patch_size)
        gru_k, pos_emb_k, ch_emb_k = jrandom.split(pred_k, 3)
        d_model_pred = config.d_inner * config.d_state
        self.step_gru = nn.GRUCell(2 * d_model_pred, d_model_pred, key=gru_k)
        self.pos_emb = jrandom.normal(
            pos_emb_k, (self.n_pred_steps, d_model_pred))
        self.ch_emb = jrandom.normal(
            ch_emb_k, (config.n_channels, d_model_pred))
        self.step_lin = Linear(
            d_model_pred, self.patch_size, key=pred_k
        )
        self.dropout = nn.Dropout(config.dropout)

    def make_component(self, config: SSMConfig.ModelConfig,  *, key: jrandom.PRNGKey):
        return SSMBlock(
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

    def _process_patches(
        self,
        x: Float[Array, "n_channels n_patches d_model"],
        ssm_state: Float[Array, "n_channels d_inner d_state"]
    ) -> Float[Array, "seq_len n_channels"]:
        dyn_components, static_components = eqx.partition(
            self.blocks, eqx.is_array)
        dyn_layer_norm1, static_layer_norm1 = eqx.partition(
            self.layer_norm1, eqx.is_array)
        dyn_layer_norm2, static_layer_norm2 = eqx.partition(
            self.layer_norm2, eqx.is_array)
        dyn_norm, static_norm = eqx.partition(self.norm, eqx.is_array)
        dyn_mlp1, static_mlp1 = eqx.partition(self.mlp1, eqx.is_array)
        dyn_mlp2, static_mlp2 = eqx.partition(self.mlp2, eqx.is_array)

        def layer_step(_carry, _dynamic):
            _hidden, _ssm_state = _carry
            _dyn_component, _dyn_layer_norm1, _dyn_layer_norm2, _dyn_norm, _dyn_mlp1, _dyn_mlp2 = _dynamic
            component = eqx.combine(_dyn_component, static_components)
            layer_norm1 = eqx.combine(_dyn_layer_norm1, static_layer_norm1)
            # layer_norm2 = eqx.combine(_dyn_layer_norm2, static_layer_norm2)
            # norm = eqx.combine(_dyn_norm, static_norm)
            # mlp1 = eqx.combine(_dyn_mlp1, static_mlp1)
            # mlp2 = eqx.combine(_dyn_mlp2, static_mlp2)

            # _upd_hidden, _norm_state = norm(_hidden)
            _upd_hidden, _upd_ssm_state = component(
                _hidden, _ssm_state
            )
            # _upd_hidden = norm.denorm(_upd_hidden, _norm_state)

            _upd_hidden = _hidden + jax.vmap(layer_norm1)(_upd_hidden)
            # _upd_hidden = _upd_hidden + \
            #     mlp2(jnn.gelu(mlp1(jax.vmap(layer_norm2)(_upd_hidden))))

            return (_upd_hidden, _upd_ssm_state), None

        step_init = (x, ssm_state)
        (hidden, ssm_state), _ = jax.lax.scan(
            layer_step, step_init, (dyn_components, dyn_layer_norm1, dyn_layer_norm2, dyn_norm, dyn_mlp1, dyn_mlp2))

        return hidden, ssm_state

    def _backbone(self, x: Array):
        _, n_channels = x.shape

        # (n_patches patch_size n_channels)
        x = self.patch(x, dim=-2)
        x = rearrange(
            x, 'n_patches patch_size n_channels -> n_channels n_patches patch_size')
        # (n_channels n_patches d_model)
        x = self.embedding(x)

        ssm_state = jnp.zeros((n_channels, self.d_inner, self.d_state))
        hidden, ssm_state = self._process_patches(x, ssm_state)

        return hidden, ssm_state

    def _predict_from_hidden(self, hidden: Float[Array, "n_channels n_patches d_model"], n_steps: int = 1) -> Float[Array, "n_channels seq_len"]:
        hidden = rearrange(
            hidden, "n_channels n_patches d_model -> n_channels (n_patches d_model)"
        )

        preds = self.step_lin(hidden)
        preds = preds.transpose(-1, -2)
        return preds

    def __call__(
        self,
        x: Float[Array, "seq_len n_channels"]
    ) -> Float[Array, "seq_len n_channels"]:

        hidden, _ = self._backbone(x)
        preds = self.step_lin(hidden)
        preds = rearrange(
            preds, "n_channels n_patches patch_size -> (n_patches patch_size) n_channels")

        return preds

    # def predict(
    #     self,
    #     x: Float[Array, "seq_len n_channels"]
    # ) -> Float[Array, "n_channels pred_len"]:
    #     hidden, ssm_state = self._backbone(x)
    #     preds = self.step_lin(hidden)
    #     # (n_channels patch_size)
    #     preds = preds[:, -1, :]

    #     @partial(jax.jit, static_argnums=1)
    #     def pred_step(_carry, _) -> Tuple[
    #         Float[Array, "n_channels n_patches d_model"],
    #         Float[Array, "n_channels d_inner d_state"],
    #     ]:
    #         _hidden, _ssm_state = _carry
    #         _upd_hidden, _upd_ssm_state = self._process_patches(
    #             _hidden, _ssm_state)

    #         _preds = self.step_lin(_upd_hidden)[:, -1, :]

    #         return (_upd_hidden, _upd_ssm_state), _preds

    #     _, next_preds = jax.lax.scan(
    #         pred_step, (hidden, ssm_state), None, self.n_pred_steps
    #     )
    #     preds = jnp.concatenate((preds[None, ...], next_preds), axis=-3)
    #     preds = rearrange(
    #         preds, "n_steps n_channels patch_size -> (n_steps patch_size) n_channels")
    #     preds = preds[:self.pred_len, :]

    #     return preds

    def predict(
        self,
        x: Float[Array, "seq_len n_channels"],
        *,
        key: jrandom.PRNGKey
    ) -> Float[Array, "n_channels pred_len"]:
        last_val = jax.lax.stop_gradient(x[-1:, :])
        x = x - last_val

        _, n_channels = x.shape
        _, ssm_state = self._backbone(x)
        ssm_state = rearrange(ssm_state, "c di ds -> c (di ds)")
        ssm_state = jnp.tile(ssm_state, (self.n_pred_steps, 1))
        emb = jnp.concatenate([
            jnp.tile(self.pos_emb, (n_channels, 1)),
            jnp.tile(self.ch_emb, (self.n_pred_steps, 1))
        ], axis=-1)

        preds = jax.vmap(self.step_gru)(emb, ssm_state)
        preds = self.dropout(preds, key=key)
        preds = self.step_lin(preds)
        preds = rearrange(
            preds, "(m c) d -> (m d) c",
            m=self.n_pred_steps, c=n_channels
        )[:self.pred_len, :]

        preds = preds + last_val

        return preds
