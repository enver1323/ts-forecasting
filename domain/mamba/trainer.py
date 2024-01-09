from functools import partial
from typing import Type, Sequence, Tuple, Optional
import os

import numpy as np
import jax
from jaxtyping import Array, Float, Integer
from jax.random import KeyArray
import jax.numpy as jnp
import equinox as eqx
import optax
from optax import GradientTransformation, OptState
from matplotlib import pyplot as plt

from domain.mamba.modules.model import Mamba
from domain.mamba.config import MambaConfig
from domain._common.trainers._base_jax_trainer import BaseJaxTrainer
from domain._common.losses.metrics_jax import mse, mae
from generics import BaseConfig


def compute_reconstruction_loss(
    model: Mamba,
    x: Float[Array, "batch context_size n_channels"],
    y: Float[Array, "batch context_size n_channels"],
) -> Tuple[Array, Tuple[Array, Array, Array]]:
    patch_size = model.patch_size

    preds = jax.vmap(model)(x)

    mse_loss = mse(preds[:, -patch_size:], y[:, :patch_size, :])
    return mse_loss, (x, y, preds)


def compute_forecasting_loss(
    model: Mamba,
    x: Float[Array, "batch context_size n_channels"],
    y: Float[Array, "batch context_size n_channels"],
) -> Tuple[Array, Tuple[Array, Array, Array]]:
    preds = jax.vmap(model.predict)(x)
    mse_loss = mse(preds, y)
    mae_loss = mae(preds, y)
    loss = mse_loss + mae_loss
    return loss, (x, y, preds, mse_loss)


class MambaTrainer(BaseJaxTrainer):
    def __init__(self, config: BaseConfig, key: KeyArray):
        self.config: MambaConfig
        self.model: Mamba
        super(MambaTrainer, self).__init__(config, key)

    @property
    def model_type(self) -> Type[Mamba]:
        return Mamba

    def get_experiment_key(self, config: MambaConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        patch_size = config.model.patch_size
        data = config.data.dataset.path.split('/')[-1].split('.')[0]
        n_blocks = config.model.n_blocks

        return f"{model_name}_{data}_({seq_len}->{pred_len}|{patch_size})x{n_blocks}_pred_loss_ssm"

    def _init_optimizers_w_states(self):
        rec_optim = optax.adamw(self.config.lr.rec)
        rec_state = rec_optim.init(eqx.filter(self.model, eqx.is_array))

        pred_optim = optax.adamw(self.config.lr.pred)
        pred_state = pred_optim.init(eqx.filter(self.model, eqx.is_array))

        return (
            # Reconstruction optimizer
            (rec_optim, rec_state),
            # Pred optimizer
            (pred_optim, pred_state),
        )

    def _get_batch_data(self, batch):
        x, y = [jnp.copy(item) for item in batch[:2]]

        return x, y

    @eqx.filter_jit
    def _step(
        self,
        model: Mamba,
        batch: Sequence[np.ndarray],
        optimizers: Optional[Sequence[Tuple[GradientTransformation, OptState]]] = None
    ):
        (rec_optim, rec_state), (pred_optim, pred_state) = optimizers or (
            (None, None), (None, None))

        x, y = self._get_batch_data(batch)

        (rec_loss, _), grads = eqx.filter_value_and_grad(
            compute_reconstruction_loss, has_aux=True)(model, x, y)

        if rec_optim is not None and rec_state is not None:
            rec_updates, rec_state = rec_optim.update(
                grads, rec_state, params=eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, rec_updates)

        x, y = self._get_batch_data(batch)
        (_, (*_, pred_mse_loss)), grads = eqx.filter_value_and_grad(
            compute_forecasting_loss, has_aux=True)(model, x, y)
        if pred_optim is not None and pred_state is not None:
            pred_updates, pred_state = pred_optim.update(
                grads, pred_state, params=eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, pred_updates)

        return (
            model, rec_loss, {"pred_loss": pred_mse_loss})

    def visualize(self, batch: Sequence[Array], filepath: str):
        self.visualize_forecasting(batch, filepath)
        # self.visualize_reconstruction(batch, filepath)

    def visualize_reconstruction(self, batch: Sequence[Array], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        b, c = 0, -1
        x, y = [jax.lax.stop_gradient(item)[b:b+1]
                for item in self._get_batch_data(batch)]

        _, (x, y, preds) = eqx.filter_jit(compute_reconstruction_loss)(x, y)
        x, y, preds = [item[0, :, c] for item in (x, y, preds)]

        plt.plot(y, label="Ground Truth", color="orange")
        plt.plot(preds, label="Prediction", color="blue")

        plt.tight_layout()
        plt.legend(loc="lower left")
        plt.savefig(filepath)
        plt.close()

    def visualize_forecasting(self, batch: Sequence[Array], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        b, c = 0, -1
        x, y = [jax.lax.stop_gradient(item)[b:b+1]
                for item in self._get_batch_data(batch)]

        _, (x, y, preds, _) = eqx.filter_jit(
            compute_forecasting_loss)(self.model, x, y)
        x, y, preds = [item[0, :, c] for item in (x, y, preds)]

        true_vals = jnp.concatenate((x, y), axis=-1)
        plt.plot(true_vals, label="Ground Truth", color="orange")
        total_len = len(true_vals)
        preds_len = len(preds)
        x_preds = jnp.arange(total_len - preds_len, total_len)
        plt.plot(x_preds, preds, label="Prediction", color="blue")

        plt.tight_layout()
        plt.legend(loc="lower left")
        plt.savefig(filepath)
        plt.close()