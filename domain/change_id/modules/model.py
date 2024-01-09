import equinox as eqx
from jaxtyping import Array, Float
import jax
from jax import numpy as jnp
from jax.random import KeyArray
from jax import random as jrandom
from typing import Tuple

from domain.change_id.config import ChangeIDConfig
from domain.change_id.modules.layers import ComponentChangeRNN


class ChangeID(eqx.Module):
    change_components: ComponentChangeRNN

    def __init__(self, config: ChangeIDConfig.ModelConfig, *, key: KeyArray):
        super(ChangeID, self).__init__()

        keys = jrandom.split(key, config.n_channels)

        def make_component(k): return ComponentChangeRNN(
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            patch_size=config.patch_size,
            stride=config.stride,
            hidden_size=config.hidden_size,
            key=k
        )
        self.change_components = eqx.filter_vmap(make_component)(keys)

    def __call__(
            self,
            x: Float[Array, "context_size n_channels"],
            x_cps: Float[Array, "context_size n_channels"],
    ) -> Tuple[
        Float[Array, "context_size n_channels"],
        Float[Array, "context_size n_channels"]
    ]:
        dynamic_components, static_components = eqx.partition(
            self.change_components, eqx.is_array
        )

        def process_channel(_x, _x_cps, _dynamic_component) -> Tuple[
            Float[Array, "context_size n_channels"],
            Float[Array, "context_size n_channels"],
        ]:
            component = eqx.combine(_dynamic_component, static_components)
            _preds, _cps = component(_x, _x_cps)
            return _preds, _cps

        x = x.transpose(-1, -2)
        x_cps = x_cps.transpose(-1, -2)

        preds, pred_cps = jax.vmap(process_channel)(x, x_cps, dynamic_components)

        preds = preds.transpose(-1, -2)
        pred_cps = pred_cps.transpose(-1, -2)

        return preds, pred_cps
