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

from domain.ssm.modules.model import SSM
from domain.ssm.config import SSMConfig
from domain._common.trainers._base_jax_trainer import BaseJaxTrainer
from domain._common.losses.metrics_jax import mse, mae
from generics import BaseConfig


# def compute_reconstruction_loss(
#     model: SSM,
#     x: Float[Array, "batch context_size n_channels"],
#     y: Float[Array, "batch context_size n_channels"]
# ) -> Tuple[Array, Tuple[Array, Array, Array]]:
#     patch_size = model.patch_size

#     preds = jax.vmap(model)(x)

#     mse_loss = mse(preds[:, -patch_size:], y[:, :patch_size, :])
#     true_vals = jnp.concatenate((x, y), axis=-2)
#     true_vals = true_vals[:, patch_size:model.seq_len + patch_size, :]
#     # mse_loss = mse(preds, true_vals)
#     return mse_loss, (x, true_vals, preds)


@eqx.filter_jit
def predict(
    model: SSM,
    x: Float[Array, "batch context_size n_channels"],
    *,
    key: Optional[Array] = None
) -> Float[Array, "batch context_size n_channels"]:
    model_w_k = partial(model.predict, key=key)
    preds = jax.vmap(model_w_k)(x)
    return preds


def compute_forecasting_loss(
    model: SSM,
    x: Float[Array, "batch context_size n_channels"],
    y: Float[Array, "batch context_size n_channels"],
    *,
    key: Optional[Array] = None
) -> Tuple[Array, Tuple[Array, Array, Array]]:
    preds = predict(model, x, key=key)
    mse_loss = mse(preds, y)
    mae_loss = mae(preds, y)

    loss = 0.5 * mse_loss + 0.5 * mae_loss
    # loss = mse_loss
    return loss, (x, y, preds, mse_loss, mae_loss)


def compute_every_step_loss(
    model: SSM,
    x: Float[Array, "batch context_size n_channels"],
    y: Float[Array, "batch context_size n_channels"],
) -> Tuple[Array, Tuple[Array, Array, Array]]:
    preds = predict(model, x)
    total_loss = 0
    patch_size = model.patch_size
    n_steps = model.n_pred_steps + 1
    for i in range(n_steps):
        ids = slice(i * patch_size, min((i + 1) * patch_size, model.pred_len))
        loss = mse(preds[:, ids, :], y[:, ids, :])
        total_loss = total_loss + loss

    mse_loss = mse(preds, y)
    total_loss = total_loss / n_steps

    return total_loss, (x, y, preds, mse_loss)


class SSMTrainer(BaseJaxTrainer):
    def __init__(self, config: BaseConfig, key: KeyArray):
        self.config: SSMConfig
        self.model: SSM
        super(SSMTrainer, self).__init__(config, key)

    @property
    def model_type(self) -> Type[SSM]:
        return SSM

    def get_experiment_key(self, config: SSMConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        patch_size = config.model.patch_size
        data = config.data.dataset.path.split('/')[-1].split('.')[0]
        n_blocks = config.model.n_blocks

        return f"{model_name}_{data}_({seq_len}->{pred_len}|{patch_size})x{n_blocks}_pred_loss_ssm"

    def _init_optimizers_w_states(self):
        lr_conf = self.config.lr
        rec_optim = optax.adamw(lr_conf.rec)
        rec_state = rec_optim.init(eqx.filter(self.model, eqx.is_array))

        n_steps_per_epoch = len(self.train_data)
        pred_scheduler = optax.exponential_decay(
            lr_conf.pred,
            n_steps_per_epoch,
            lr_conf.decay,
            lr_conf.n_warmup_epochs,
            staircase=True)
        pred_optim = optax.adamw(pred_scheduler)
        pred_state = pred_optim.init(eqx.filter(self.model, eqx.is_array))

        return (
            # Reconstruction optimizer
            # (rec_optim, rec_state),
            # Pred optimizer
            (pred_optim, pred_state, pred_scheduler),
        )

    def _get_batch_data(self, batch):
        x, y, = batch[:2]
        return x, y

    @eqx.filter_jit
    def _step(
        self,
        model: SSM,
        batch: Sequence[np.ndarray],
        optimizers: Optional[Sequence[Tuple[GradientTransformation, OptState]]] = None,
        *,
        key: Optional[KeyArray] = None,
    ):
        # (rec_optim, rec_state), (pred_optim, pred_state) = optimizers or (
        #     (None, None), (None, None))
        (pred_optim, pred_state, _), = optimizers or (
            (None, None, None),)

        x, y = self._get_batch_data(batch)

        # (rec_loss, _), grads = eqx.filter_value_and_grad(
        #     compute_reconstruction_loss, has_aux=True)(model, x, y)

        # if rec_optim is not None and rec_state is not None:
        #     rec_updates, rec_state = rec_optim.update(
        #         grads, rec_state, params=eqx.filter(model, eqx.is_array)
        #     )
        #     model = eqx.apply_updates(model, rec_updates)

        forecasting_loss_fn = partial(compute_forecasting_loss, key=key)
        (forecasting_loss, (*_, pred_mse_loss, pred_mae_loss)), grads = eqx.filter_value_and_grad(
            forecasting_loss_fn, has_aux=True)(model, x, y)
        if pred_optim is not None and pred_state is not None:
            pred_updates, pred_state = pred_optim.update(
                grads, pred_state, params=eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, pred_updates)

        # total_loss = rec_loss + forecasting_loss
        total_loss = forecasting_loss

        return (model, total_loss, {"loss": total_loss, "mse": pred_mse_loss, "mae": pred_mae_loss})

    def visualize(self, batch: Sequence[Array], filepath: str):
        self.visualize_forecasting(batch, filepath)

    def visualize_forecasting(self, batch: Sequence[Array], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        b, c = 0, -1
        x, y = [jax.lax.stop_gradient(item)[b:b+1]
                for item in self._get_batch_data(batch)]

        preds = predict(self.model, x)
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
