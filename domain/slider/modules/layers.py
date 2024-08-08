from abc import ABC
from equinox import nn
import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jrandom
import jax.numpy as jnp

from domain._common.modules.decomposition_jax import SeriesDecomposition, MovingAverage


class BackboneSlider(eqx.Module, ABC):
    trend_proj: nn.Linear
    seasonality_proj: nn.Linear
    series_decomposition: SeriesDecomposition

    kernel_size: int = eqx.static_field()
    stride: int = eqx.static_field()

    def __init__(
        self,
        c_in: int,
        kernel_size: int,
        decomposition_kernel_size: int,
        stride: int = 1,
        out_dim: int = None,
        *,
        key: jrandom.PRNGKey
    ):
        (
            trend_proj_k,
            seasonality_proj_k,
        ) = jrandom.split(key, 2)

        super(BackboneSlider, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.trend_proj = nn.Linear(c_in, out_dim, key=trend_proj_k)
        self.seasonality_proj = nn.Linear(
            c_in, out_dim, key=seasonality_proj_k)
        self.series_decomposition = SeriesDecomposition(MovingAverage(decomposition_kernel_size))

    def _generate_slices(self, path, carry, slice_idx):
        start_indices = (slice_idx * self.stride, 0)
        slice_shape = (self.kernel_size, path.shape[-1])
        path_slice = jax.lax.dynamic_slice(path, start_indices, slice_shape)

        return carry, path_slice

    def _slide_over_slices(self, trend, proj_module):
        return jax.vmap(proj_module)(trend)

    def _decompose_path(self, path: jnp.ndarray) -> jnp.ndarray:
        seq_len = path.shape[-2]

        trend, seasonality = self.series_decomposition(path)

        slices_num = (seq_len - self.kernel_size) // self.stride + 1
        slices_range = jnp.arange(0, slices_num)

        _, trend_slices = jax.lax.scan(
            lambda *args: self._generate_slices(trend, *args), None, slices_range)

        _, seasonality_slices = jax.lax.scan(
            lambda *args: self._generate_slices(seasonality, *args), None, slices_range)

        trend_slices = jax.vmap(lambda trend_slice: self._slide_over_slices(
            trend_slice, self.trend_proj))(trend_slices)

        seasonality_slices = jax.vmap(lambda seasonality_slice: self._slide_over_slices(
            seasonality_slice, self.seasonality_proj))(seasonality_slices)

        return trend_slices + seasonality_slices

class LinearSlider(BackboneSlider):
    fwd_cum_nets: list
    back_cum_nets: list

    def __init__(
        self,
        c_in: int,
        kernel_size: int,
        decomposition_kernel_size: int,
        stride: int = 1,
        out_dim: int = None,
        *,
        key: jrandom.PRNGKey
    ):
        (
            fwd_cum_k,
            back_cum_k,
        ) = jrandom.split(key, 2)

        super(LinearSlider, self).__init__(
            c_in,
            kernel_size,
            decomposition_kernel_size,
            stride,
            out_dim,
            key=key
        )
        self.fwd_cum_nets = self._build_cum_layers(out_dim=out_dim, key=fwd_cum_k)
        self.back_cum_nets = self._build_cum_layers(out_dim=out_dim, key=back_cum_k)

    def _build_cum_layers(self, out_dim, *, key: jrandom.PRNGKey):
        (lin_proj_k, cum_proj_k, combination_k) = jrandom.split(key, 3)
        return [
            nn.Linear(out_dim, 4 * out_dim, key=lin_proj_k),
            nn.Linear(out_dim, 4 * out_dim, key=cum_proj_k),
            nn.Linear(4 * out_dim, out_dim, key=combination_k)
        ]

    def _combine_slices(self, combined: jnp.ndarray, path_slice: jnp.ndarray, cum_modules):
        lin_proj, cum_proj, combination = cum_modules
        
        path_slice = jnn.gelu(jax.vmap(lin_proj)(path_slice))
        combined = jnn.gelu(jax.vmap(cum_proj)(combined))

        combined = jax.vmap(combination)(path_slice * combined)
        combined = jnn.tanh(combined)

        return combined, None

    def __call__(self, path: jnp.ndarray) -> jnp.ndarray:
        slices = self._decompose_path(path)

        fwd_cum, _ = jax.lax.scan(
            lambda *args: self._combine_slices(*args, self.fwd_cum_nets), slices[0], slices[1:])

        back_cum, _ = jax.lax.scan(
            lambda *args: self._combine_slices(*args, self.back_cum_nets), slices[-1], slices[-2::-1])

        return jnp.concatenate((fwd_cum, back_cum), axis=-1)

class GRUSlider(BackboneSlider):
    fwd_lin_in: nn.Linear
    fwd_lin_hid: nn.Linear

    back_lin_in: nn.Linear
    back_lin_hid: nn.Linear

    def __init__(
        self,
        c_in: int,
        kernel_size: int,
        decomposition_kernel_size: int,
        stride: int = 1,
        out_dim: int = None,
        *,
        key: jrandom.PRNGKey
    ):
        (
            lin_in_k,
            lin_hid_k,
            back_in_k,
            back_hid_k,
        ) = jrandom.split(key, 4)

        super(GRUSlider, self).__init__(
            c_in,
            kernel_size,
            decomposition_kernel_size,
            stride,
            out_dim,
            key=key
        )
        
        self.fwd_lin_hid = nn.Linear(out_dim, out_dim, key=lin_in_k)
        self.fwd_lin_in = nn.Linear(out_dim, out_dim, key=lin_hid_k)

        self.back_lin_hid = nn.Linear(out_dim, out_dim, key=back_in_k)
        self.back_lin_in = nn.Linear(out_dim, out_dim, key=back_hid_k)

    def _combine_slices(self, lin_hid, lin_in, carry, item):
        hid = jax.vmap(lin_hid)(carry)
        cur = jax.vmap(lin_in)(item)
        hid = jnn.tanh(hid + cur)
        return hid, None

    def __call__(self, path: jnp.ndarray) -> jnp.ndarray:
        slices = self._decompose_path(path)

        fwd_hid, _ = jax.lax.scan(lambda *args: self._combine_slices(self.fwd_lin_hid, self.fwd_lin_in, *args), slices[0], slices[1:])
        back_hid, _ = jax.lax.scan(lambda *args: self._combine_slices(self.back_lin_hid, self.back_lin_in, *args), slices[-1], slices[-2::-1])

        return jnp.concatenate((fwd_hid, back_hid), axis=-1)

class LSTMSlider(BackboneSlider):
    fwd_w_list: list
    back_w_list: list
    fwd_u_list: list
    back_u_list: list

    out_dim: int = eqx.static_field()

    def __init__(
        self,
        c_in: int,
        kernel_size: int,
        decomposition_kernel_size: int,
        stride: int = 1,
        out_dim: int = None,
        *,
        key: jrandom.PRNGKey
    ):
        (
            fwd_w_k,
            fwd_u_k,
            back_w_k,
            back_u_k
        ) = jrandom.split(key, 4)

        super(LSTMSlider, self).__init__(
            c_in,
            kernel_size,
            decomposition_kernel_size,
            stride,
            out_dim,
            key=key
        )

        self.out_dim = out_dim

        self.fwd_w_list = self._build_lstm_block(out_dim, key=fwd_w_k)
        self.fwd_u_list = self._build_lstm_block(out_dim, key=fwd_u_k)
        self.back_w_list = self._build_lstm_block(out_dim, key=back_w_k)
        self.back_u_list = self._build_lstm_block(out_dim, key=back_u_k)

    def _build_lstm_block(self, dim, *, key):
        (
            forget_k,
            input_k,
            output_k,
            cell_k
        ) = jrandom.split(key, 4)

        return [
            nn.Linear(dim, dim, key=forget_k),  # Forget gate
            nn.Linear(dim, dim, key=input_k),   # Input gate
            nn.Linear(dim, dim, key=output_k),  # Output gate
            nn.Linear(dim, dim, key=cell_k),    # Cell gate
        ]

    def _combine_slices(self, hid_module_list: list, slice_module_list, hidden, path_slice: jnp.ndarray):
        hidden, cell = hidden
        
        f_t = jnn.sigmoid(jax.vmap(hid_module_list[0])(hidden) + jax.vmap(slice_module_list[0])(path_slice))
        i_t = jnn.sigmoid(jax.vmap(hid_module_list[1])(hidden) + jax.vmap(slice_module_list[1])(path_slice))
        o_t = jnn.sigmoid(jax.vmap(hid_module_list[2])(hidden) + jax.vmap(slice_module_list[2])(path_slice))
        _c_t = jnn.tanh(jax.vmap(hid_module_list[3])(hidden) + jax.vmap(slice_module_list[3])(path_slice))

        cell_upd = jnn.sigmoid(f_t * cell + i_t * _c_t)
        hidden_upd = o_t * jnn.tanh(cell_upd)

        return hidden_upd, cell_upd
    
    def __call__(self, path):
        slices = self._decompose_path(path)
        reverse_slices = slices[::-1]
        seq_len = slices.shape[0]
        init_state = jnp.zeros_like(slices[0]), jnp.zeros_like(slices[0])

        fwd_hid, _ = jax.lax.fori_loop(
            1, seq_len,
            lambda idx, carry: self._combine_slices(self.fwd_w_list, self.fwd_u_list, carry, slices[idx]),
            init_state
        )

        init_state = jnp.zeros_like(slices[0]), jnp.zeros_like(slices[0])
        back_hid, _ = jax.lax.fori_loop(
            1, seq_len,
            lambda idx, carry: self._combine_slices(self.back_w_list, self.back_u_list, carry, reverse_slices[idx]),
            init_state
        )

        return jnp.concatenate((fwd_hid, back_hid), axis=-1)

class Predictor(eqx.Module):
    lin_seq: nn.Linear
    out_feat: nn.Linear
    lin_feat: nn.Linear
    out_seq: nn.Linear

    def __init__(
        self,
        in_feat_size: int, hid_feat_size: int, out_feat_size: int,
        in_seq_size: int, hid_seq_size: int, out_seq_size: int,
        *,
        key: jrandom.PRNGKey
    ):
        (
            hid_feat_k,
            out_feat_k,
            hid_seq_k,
            out_seq_k
        ) = jrandom.split(key, 4)

        self.lin_feat = nn.Linear(in_feat_size, hid_feat_size, key=hid_feat_k)
        self.out_feat = nn.Linear(hid_feat_size, out_feat_size, key=out_feat_k)

        self.lin_seq = nn.Linear(
            in_seq_size, hid_seq_size, key=hid_seq_k)
        self.out_seq = nn.Linear(hid_seq_size, out_seq_size, key=out_seq_k)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.vmap(self.lin_feat)(x)
        x = jnn.gelu(x)
        x = jax.vmap(self.out_feat)(x)
        x = jnn.gelu(x)

        x = x.transpose(-1, -2)
        x = jax.vmap(self.lin_seq)(x)
        x = jnn.gelu(x)
        x = jax.vmap(self.out_seq)(x)

        x = x.transpose(-1, -2)

        return x