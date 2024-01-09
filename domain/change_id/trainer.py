from typing import Type, Sequence, Tuple, Optional
import os

import numpy as np
import jax
from jaxtyping import Array, Float
from jax.random import KeyArray
import jax.numpy as jnp
import equinox as eqx
from optax import adamw, GradientTransformation, OptState
from matplotlib import pyplot as plt

from domain.change_id.modules.model import ChangeID
from domain.change_id.config import ChangeIDConfig
from domain._common.trainers._base_jax_trainer import BaseJaxTrainer
from domain._common.losses.metrics_jax import mse, bce_loss
from generics import BaseConfig


def compute_data_loss(
    model: ChangeID,
    x: Float[Array, "batch context_size n_channels"],
    x_cps: Float[Array, "batch context_size n_channels"],
    y: Float[Array, "batch context_size n_channels"],
    y_cps: Float[Array, "batch context_size n_channels"]
) -> Tuple[Array, Tuple[Array, Array]]:
    preds, _ = jax.vmap(model)(x, x_cps)
    offset = model.change_components.patch_size
    true_vals = jnp.concatenate((x[:, offset:, :], y), axis=-2)

    mse_loss = mse(preds, true_vals)
    pred_mse = mse(preds[:, -model.change_components.pred_len:, :], y)

    return mse_loss, (mse_loss, pred_mse)


def compute_cp_loss(
    model: ChangeID,
    x: Float[Array, "batch context_size n_channels"],
    x_cps: Float[Array, "batch context_size n_channels"],
    y: Float[Array, "batch context_size n_channels"],
    y_cps: Float[Array, "batch context_size n_channels"]
) -> Array:
    all_cps = jnp.concatenate((x_cps, y_cps), axis=-2)
    _, pred_cps = jax.vmap(model)(x, x_cps)

    component = model.change_components
    indices = jnp.arange(0, component.n_patches)

    def _extract(idx):
        return component.slice_patch(all_cps, idx, -2).any(-2).astype(int)

    true_cps = jax.vmap(_extract)(indices)
    true_cps = true_cps.transpose(1, 0, 2)

    cp_loss = bce_loss(pred_cps, true_cps)

    return cp_loss


class ChangeIDTrainer(BaseJaxTrainer):
    def __init__(self, config: BaseConfig, key: KeyArray):
        self.config: ChangeIDConfig
        super(ChangeIDTrainer, self).__init__(config, key)
        self.model: ChangeID = self.model_type(self.config.model, key=self.key)

    @property
    def model_type(self) -> Type[ChangeID]:
        return ChangeID

    def get_experiment_key(self, config: ChangeIDConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.seq_len
        patch_size = config.model.patch_size
        stride = config.model.stride
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len}:{patch_size}/{stride})"

    def _init_optimizers_w_states(self):
        data_optim = adamw(self.config.lr.init)
        data_state = data_optim.init(eqx.filter(self.model, eqx.is_array))

        cp_optim = adamw(self.config.lr.init)
        cp_state = cp_optim.init(eqx.filter(self.model, eqx.is_array))

        return (
            # Global optimizer
            (data_optim, data_state),
            # CP optimizer
            (cp_optim, cp_state),
        )

    @eqx.filter_jit
    def _step(
        self,
        model: ChangeID,
        batch: Sequence[np.ndarray],
        optimizers: Optional[Sequence[Tuple[GradientTransformation, OptState]]] = None
    ):
        (data_optim, data_state), (cp_optim, cp_state) = optimizers or (
            (None, None), (None, None))

        x, y, x_cps, y_cps, *_ = batch

        (data_loss, (mse_loss, pred_mse)), grads = eqx.filter_value_and_grad(
            compute_data_loss, has_aux=True)(model, x, x_cps, y, y_cps)

        if data_optim is not None and data_state is not None:
            data_updates, data_state = data_optim.update(
                grads, data_state, params=eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, data_updates)

        cp_loss, grads = eqx.filter_value_and_grad(
            compute_cp_loss)(model, x, x_cps, y, y_cps)
        if cp_optim is not None and cp_state is not None:
            cp_updates, cp_state = data_optim.update(
                grads, cp_state, params=eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, cp_updates)

        return (
            model, (
                pred_mse,
                mse_loss,
                cp_loss
            ), {
                "DataLoss": data_loss,
                "TotalMSE": mse_loss,
                "PredMSE": pred_mse,
                "CPLoss": cp_loss,
            })

    def visualize(self, batch: Sequence[Array], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        b, c = 0, -1
        batch = [jax.lax.stop_gradient(item)[b, :, :] for item in batch]
        x, y, x_cps, y_cps, *_ = batch

        preds, pred_cps = eqx.filter_jit(self.model)(x, x_cps)

        preds = preds[:, c]
        pred_cps = pred_cps[:, c]
        x, y, x_cps, y_cps = [item[:, c] for item in (x, y, x_cps, y_cps)]

        x = jnp.concatenate((x, y))
        cps = jnp.concatenate((x_cps, y_cps))

        min_y = min(x.min().item(), preds.min().item())
        max_y = max(x.max().item(), preds.max().item())

        offset = self.config.model.patch_size

        x = x[offset:]
        cps = cps[offset:]

        plt.plot(x, label="Ground Truth", color="orange")
        plt.plot(preds, label="Prediction", color="blue")

        cps = cps.nonzero()
        plt.vlines(x=cps, ymin=min_y, ymax=max_y,
                   linestyles='dashed', color='orange', label='GT cps')

        stride, patch_size = self.config.model.stride, self.config.model.patch_size
        pred_cps = (pred_cps > 0.5).nonzero()
        pred_cps = [pred_cp * stride + patch_size // 2 for pred_cp in pred_cps]
        plt.vlines(x=pred_cps, ymin=min_y, ymax=max_y,
                   color='blue', label='Pred cps', linestyles='dashed')

        plt.tight_layout()
        plt.legend(loc="lower left")
        plt.savefig(filepath)
        plt.close()