import equinox as eqx
from equinox import nn, static_field
from jaxtyping import Float, Array
import jax
from jax import (
    numpy as jnp,
    nn as jnn,
    random as jrandom
)
from typing import Tuple
from domain._common.losses.metrics_jax import cosine_similarity


class ComponentChangeRNN(eqx.Module):
    in_to_loc: nn.Linear
    in_to_stat: nn.Linear
    in_to_glob: nn.Linear
    loc_to_loc: nn.Linear
    loc_to_stat: nn.Linear
    glob_to_glob: nn.Linear
    hid_to_out: nn.Linear

    seq_len: int = static_field()
    pred_len: int = static_field()
    patch_size: int = static_field()
    stride: int = static_field()
    hidden_size: int = static_field()
    n_patches: int = static_field()
    n_seq_patches: int = static_field()
    n_pred_patches: int = static_field()

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        stride: int,
        hidden_size: int,
        *,
        key: jrandom.KeyArray
    ):
        super(ComponentChangeRNN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.stride = stride
        self.hidden_size = hidden_size

        i2lk, i2sk, i2gk, l2lk, l2sk, g2gk, h2ok = jrandom.split(key, 7)

        self.in_to_loc = nn.Linear(patch_size, hidden_size, key=i2lk)
        self.in_to_stat = nn.Linear(patch_size - 1, hidden_size, key=i2sk)
        self.in_to_glob = nn.Linear(patch_size, hidden_size, key=i2gk)

        self.loc_to_loc = nn.Linear(hidden_size, hidden_size, key=l2lk)
        self.loc_to_stat = nn.Linear(hidden_size, hidden_size, key=l2sk)
        self.glob_to_glob = nn.Linear(hidden_size, hidden_size, key=g2gk)

        self.hid_to_out = nn.Linear(2 * hidden_size, self.stride, key=h2ok)

        self.n_seq_patches = (self.seq_len - self.patch_size) // self.stride
        self.n_pred_patches = self.pred_len // self.stride

        self.n_patches = self.n_seq_patches + self.n_pred_patches

    def slice_patch(self, datum: Float[Array, "context_size"], idx: int, dim: int = 0) -> Float[Array, "patch_size"]:
        return jax.lax.dynamic_slice_in_dim(datum, idx * self.stride, self.patch_size, dim)

    def __call__(
        self,
        x: Float[Array, "seq_len"],
        cps: Float[Array, "seq_len"],
    ) -> Tuple[Float[Array, "context_size"], Float[Array, "num_patches"]]:
        patch = self.slice_patch(x, 0)
        loc_hid = jnn.gelu(self.in_to_loc(patch))
        glob_hid = jnn.gelu(self.in_to_glob(patch))

        def process_seq_patch(carry, idx):
            loc_hid, glob_hid = carry
            patch = self.slice_patch(x, idx)
            in_stat = self.slice_patch(cps, idx).any(axis=-1).astype(int)

            in_to_stat_out = self.in_to_stat(patch[1:] - patch[:-1])
            stat = cosine_similarity(
                in_to_stat_out, self.loc_to_stat(loc_hid), axis=-1
            )
            glob_hid = jnn.gelu(self.in_to_glob(patch) + self.glob_to_glob(glob_hid)) * (1 - in_stat) \
                + glob_hid * (in_stat)
            loc_hid = jnn.gelu(self.loc_to_loc(loc_hid) + self.in_to_loc(patch)) * in_stat \
                + glob_hid * (1 - in_stat)
            outputs = self.hid_to_out(jnp.concatenate((loc_hid, glob_hid), axis=-1))

            return (loc_hid, glob_hid), (outputs, stat)

        _, (out_seq, out_seq_cps) = jax.lax.scan(
            process_seq_patch, (loc_hid, glob_hid), jnp.arange(0, self.n_seq_patches)
        )

        out_seq = out_seq.reshape(self.n_seq_patches * self.stride)
        out_seq_cps = out_seq_cps.reshape(self.n_seq_patches)

        out_pred = x[-self.patch_size:].copy()

        def process_pred_patch(carry, idx):
            (patch, loc_hid, glob_hid) = carry

            in_to_stat_out = self.in_to_stat(patch[1:] - patch[:-1])

            stat = cosine_similarity(
                in_to_stat_out, self.loc_to_stat(loc_hid), axis=-1
            )
            glob_hid = jnn.gelu(self.in_to_glob(patch) + self.glob_to_glob(glob_hid)) * (1 - stat) \
                + glob_hid * (stat)
            loc_hid = jnn.gelu(self.loc_to_loc(loc_hid) + self.in_to_loc(patch)) * stat \
                + glob_hid * (1 - stat)
            outputs = self.hid_to_out(jnp.concatenate((loc_hid, glob_hid), axis=-1))


            out_pred = jnp.concatenate((patch, outputs), axis=0)
            out_pred = out_pred[-self.patch_size:]

            return (out_pred, loc_hid, glob_hid), (outputs, stat)

        _, (out_pred, out_pred_cps) = jax.lax.scan(
            process_pred_patch, (out_pred, loc_hid, glob_hid), jnp.arange(0, self.n_pred_patches)
        )
        out_pred = out_pred.reshape(self.n_pred_patches * self.stride)
        out = jnp.concatenate((out_seq, out_pred), axis=0)
        
        out_pred_cps = out_pred_cps.reshape(self.n_pred_patches)
        out_cps = jnp.concatenate((out_seq_cps, out_pred_cps), axis=0)

        return out, out_cps

    def get_stat_parameters(self):
        stat_layers = {
            self.in_to_hid, self.loc_hid_change, self.glob_hid_change, self.hid_to_stat, self.stat_classifier
        }

        return eqx.filter(self, lambda x: x in stat_layers)
