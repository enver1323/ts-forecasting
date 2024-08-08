from typing import Type, Sequence, Tuple, Optional
import os

import numpy as np
import jax
from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx
import optax
from optax import GradientTransformation, OptState
from matplotlib import pyplot as plt

from domain.slider.modules.model import SlidingPredictor
from domain.slider.config import SliderConfig
from domain._common.trainers._base_jax_trainer import BaseJaxTrainer
from domain._common.losses.metrics_jax import mse, mae
from generics import BaseConfig


@eqx.filter_jit
def predict(
    model: SlidingPredictor,
    x: Float[Array, "batch seq_len n_channels"]
) -> Float[Array, "batch context_size n_channels"]:
    preds = jax.vmap(model)(x)
    return preds


def compute_loss(
    model: SlidingPredictor,
    x: Float[Array, "batch seq_len n_channels"],
    y: Float[Array, "batch pred_len n_channels"]
) -> Tuple[Array, Tuple[Array, Array, Array]]:
    preds = predict(model, x)
    mse_loss = mse(preds, y)
    mae_loss = mae(preds, y)

    loss = mse_loss
    return loss, (y, preds, mse_loss, mae_loss)


class SliderTrainer(BaseJaxTrainer):
    def __init__(self, config: BaseConfig, key: Array):
        self.config: SliderConfig
        self.model: SlidingPredictor
        super(SliderTrainer, self).__init__(config, key)

    @property
    def model_type(self) -> Type[SlidingPredictor]:
        return SlidingPredictor

    def get_experiment_key(self, config: SliderConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        patch_size = config.model.patch_size
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len}|{patch_size})"

    def _init_optimizers_w_states(self):
        pred_optim = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.config.lr.init, weight_decay=4e-3)
        )
        pred_state = pred_optim.init(eqx.filter(self.model, eqx.is_array))

        return (pred_optim, pred_state)

    def _get_batch_data(self, batch):
        x, y = batch[:2]
        return x, y

    @eqx.filter_jit
    def _step(
        self,
        model: SlidingPredictor,
        batch: Sequence[np.ndarray],
        optimizers: Optional[Sequence[Tuple[GradientTransformation, OptState]]] = None,
        *,
        key: Array = None
    ):
        pred_optim, pred_state = optimizers or (None, None)

        x, y = self._get_batch_data(batch)

        (loss, (*_, pred_mse_loss, pred_mae_loss)), grads = eqx.filter_value_and_grad(
            compute_loss, has_aux=True)(model, x, y)
        if pred_optim is not None and pred_state is not None:
            pred_updates, pred_state = pred_optim.update(
                grads, pred_state, params=eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, pred_updates)

        return (model, loss, {"pred_loss": pred_mse_loss, "pred_mae": pred_mae_loss})

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
