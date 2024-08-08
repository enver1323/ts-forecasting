import os
from typing import Optional, Sequence, Type, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

from domain.rnn_dec.modules.model import RNNDec
from domain.rnn_dec.modules.layers import StatsPred
from domain.rnn_dec.config import RNNDecConfig
from domain._common.trainers._base_torch_trainer import BaseTorchTrainer
from domain._common.losses.metrics_torch import mse, mae, kl_divergence, log_cosh_loss
from domain._common.losses.softdtw_loss.soft_dtw_torch import SoftDTW
from einops import rearrange


class RNNDecTrainer(BaseTorchTrainer):
    def __init__(self, config: RNNDecConfig, device: torch.device):
        self.config: RNNDecConfig
        self.model: StatsPred

        super(RNNDecTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[StatsPred]:
        return StatsPred

    def get_experiment_key(self, config: RNNDecConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        patch_len = config.model.patch_len
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len}|{patch_len})"

    def _get_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        return (
            torch.optim.Adam(self.model.parameters(), self.config.lr.init),
        )

    def _predict_stats(self, batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, y, x_date, y_date, *_ = batch
        y = y[:, -self.config.model.pred_len:, :]
        y_date = y_date[:, -self.config.model.pred_len:, :]
        (x_mean, x_std), hid = self.model.encode(x, x_date)
        y_mean, y_std = self.model.decode(hid, y_date)
        return (x_mean, x_std, y_mean, y_std), (x, y)

    def _predict(self, batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, y, *_ = batch
        y = y[:, -self.config.model.pred_len:, :]

        x_mean, x_std, y_mean, y_std = [
            stat.repeat_interleave(self.config.model.patch_len, dim=-1)
                .transpose(-2, -1)
            for stat in self._predict_stats(batch)[0]
        ]
        epsilon = 1e-8
        x_upd = (x - x_mean) / (x_std + epsilon)
        preds = self.model(x_upd)
        preds = (preds * (y_std + epsilon)) + y_mean

        return (preds, (x_mean, x_std, y_mean, y_std)), (x, y)

    def _compute_pred_loss(self, batch) -> Tuple[Tuple[Tensor], Tensor, Dict[str, Any]]:
        (preds, *_), (x, y) = self._predict(batch)
        pred_mae = mae(preds, y)
        pred_mse = mse(preds, y)
        pred_loss = pred_mae + pred_mse

        return (pred_loss,), {
            "pred_mse": pred_mse.item(),
            "pred_mae": pred_mae.item(),
        }

    def _compute_stat_loss(self, batch) -> Tuple[Tuple[Tensor], Tensor, Dict[str, Any]]:
        preds, (x, y) = self._predict_stats(batch)

        with torch.no_grad():
            x_patch = rearrange(
                x.detach(), 'b (n w) c -> b c n w', w=self.model.patch_len)
            x_true_mean = x_patch.mean(-1)
            x_true_std = x_patch.std(-1)
            y_patch = rearrange(
                y.detach(), 'b (n w) c -> b c n w', w=self.model.patch_len)
            y_true_mean = y_patch.mean(-1)
            y_true_std = y_patch.std(-1)

            y_true_diff = y_true_mean[:, :, 1:] - y_true_mean[:, :, :-1]
            y_true_diff = y_true_diff / \
                (y_true_diff.norm(2, dim=-1, keepdim=True) + 1e-8)

        (x_pred_mean, x_pred_std, y_pred_mean, y_pred_std) = preds

        y_mse = mse(y_pred_mean, y_true_mean) + mse(y_pred_std, y_true_std)
        x_mse = mse(x_pred_mean, x_true_mean) + mse(x_pred_std, x_true_std)

        y_mae = mae(y_pred_mean, y_true_mean) + mae(y_pred_std, y_true_std)
        x_mae = mae(x_pred_mean, x_true_mean) + mae(x_pred_std, x_true_std)

        stat_loss = y_mse + y_mae + x_mse + x_mae
        
        metrics = {
            "Loss": stat_loss.item(),
            "x_mse": x_mse.item(),
            "y_mse": y_mse.item(),
            "x_mae": x_mae.item(),
            "y_mae": y_mae.item(),
        }

        return (stat_loss,), metrics

    def _scheduler_step(
        self,
        schedulers: Sequence[torch.optim.lr_scheduler.LRScheduler],
        optimizers: Sequence[torch.optim.Optimizer],
        loss: torch.Tensor,
        epoch: int
    ):
        lr = self.config.lr.init
        decay = self.config.lr.decay
        if epoch > 2:
            lr = lr * (decay ** ((epoch - 2) // 1))

        for optim in optimizers:
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def _step(self, batch: Sequence[torch.Tensor], optimizers: Optional[Sequence[torch.optim.Optimizer]] = None, epoch: int = None):
        stat_optim, = optimizers or (None,)

        if stat_optim is not None:
            stat_optim.zero_grad()

        (loss,), aux_data = self._compute_stat_loss(batch)

        if stat_optim is not None:
            loss.backward()
            stat_optim.step()
        loss = loss.detach()

        return (loss, ), aux_data

    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with torch.no_grad():
            (preds, aux), (x, y), *_ = self._predict_stats(batch)
            x = x.detach().cpu()[0, :, -1]
            y = y.detach().cpu()[0, :, -1]
            preds = preds.detach().cpu()[0, :, -1]

            gt = torch.concat([x, y])
            preds = torch.concat([x, preds])

            patch_len = self.model.patch_len

            x_patch = rearrange(x, '(n w) -> n w', w=patch_len)
            x_mean = x_patch.mean(-1)
            x_std = x_patch.std(-1)
            y_patch = rearrange(y, '(n w) -> n w', w=patch_len)
            y_mean = y_patch.mean(-1)
            y_std = y_patch.std(-1)

            true_mean = torch.concat(
                [x_mean, y_mean]).repeat_interleave(patch_len)
            true_std = torch.concat(
                [x_std, y_std]).repeat_interleave(patch_len)

            pred_x_mean, pred_x_std, pred_y_mean, pred_y_std = [
                datum.detach().cpu()[0, :, -1]
                for datum in aux
            ]

            pred_mean = torch.concat([pred_x_mean, pred_y_mean])
            pred_std = torch.concat([pred_x_std, pred_y_std])

            input_len = len(pred_x_mean)

            # matplotlib.rcParams.update({'font.size': 15})
            plt.plot(preds, label='Predictions', linewidth=2)
            plt.plot(gt, label='Ground Truth', linewidth=2)
            plt.vlines(input_len, -2, 0)
            plt.plot(true_mean, label='true_mean', linewidth=2)
            plt.plot(pred_mean, label='pred_mean', linewidth=2)
            # plt.plot(true_std, label='true_std', linewidth=2)
            # plt.plot(pred_std, label='pred_std', linewidth=2)

            plt.tight_layout()
            plt.legend()
            plt.savefig(filepath)
            plt.close()

    def test(self, *args):
        self.early_stopping.load_checkpoint(
            self.model, f"checkpoints/{self.experiment_key}")

        self.evaluate(self.test_data, 'test', os.path.join(
            self.PLOT_PATH, self.experiment_key), self.config.n_epochs)

        with open(self.RESULTS_PATH, 'a+') as file:
            self.log.write_text_all(file)
