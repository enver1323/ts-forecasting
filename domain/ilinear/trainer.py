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

from domain._common.trainers._base_jax_trainer import BaseJaxTrainer
from domain.ilinear.modules.model import ILinear
from domain.ilinear.config import ILinearConfig
from domain._common.losses.metrics_jax import mse
from generics import BaseConfig


def compute_loss(model: ILinear, x: Float[Array, "batch context_size n_channels"], y: Float[Array, "batch context_size n_channels"]) -> Tuple[Array, Tuple[Array, Array]]:
    preds = jax.vmap(model)(x)
    true_vals = y[:, -model.pred_len:, :]

    # soft_dtw = SoftDTW()
    # soft_dtw_loss = soft_dtw(preds, true_vals).mean()
    mse_loss = mse(preds, true_vals)

    # mse_scale = 100.0
    # data_loss = soft_dtw_loss + mse_scale * mse_loss

    # return data_loss, (soft_dtw_loss, mse_loss)
    return mse_loss, (mse_loss, mse_loss)


class ILinearTrainer(BaseJaxTrainer):
    def __init__(self, config: BaseConfig, key: KeyArray):
        self.config: ILinearConfig
        super(ILinearTrainer, self).__init__(config, key)
        self.model: ILinear = self.model_type(self.config.model, key=self.key)

    @property
    def model_type(self) -> Type[ILinear]:
        return ILinear

    def get_experiment_key(self, config: ILinearConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len})"

    def _init_optimizers_w_states(self):
        data_optim = adamw(self.config.lr.init)
        data_state = data_optim.init(eqx.filter(self.model, eqx.is_array))

        return (
            # Global optimizer
            (data_optim, data_state)
        )

    @eqx.filter_jit
    def _step(
        self,
        model: ILinear,
        batch: Sequence[np.ndarray],
        optimizers: Optional[Sequence[Tuple[GradientTransformation, OptState]]] = None
    ):
        data_optim, data_state = optimizers or (None, None)

        x, y, *_ = batch

        (data_loss, (soft_dtw, mse_loss)), grads = eqx.filter_value_and_grad(
            compute_loss, has_aux=True)(model, x, y)

        if data_optim is not None and data_state is not None:
            data_updates, data_state = data_optim.update(
                grads, data_state, params=eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, data_updates)

        return (
            model, (
                data_loss,
            ), {
                "DataLoss": data_loss,
                "MSE": mse_loss,
                "SoftDTW": soft_dtw,
            })

    def visualize(self, batch: Sequence[Array], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        x, y, *_ = batch
        x = jax.lax.stop_gradient(x)
        y = jax.lax.stop_gradient(y)

        c = -1
        y = y[0, :, c]
        x = x[0, :, :]
        preds = eqx.filter_jit(self.model)(x)  # Context, C

        x = x[:, c]
        preds = preds[:, c]

        y = y[-self.model.pred_len:]
        preds = preds[-self.model.pred_len:]

        y = jnp.concatenate([x, y])
        preds = jnp.concatenate([x, preds])

        plt.plot(y, label="Ground Truth", color="blue")
        plt.plot(preds, label="Prediction", color="orange")

        plt.tight_layout()
        plt.legend(loc="lower left")
        plt.savefig(filepath)
        plt.close()
