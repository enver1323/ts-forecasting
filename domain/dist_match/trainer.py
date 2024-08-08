import os
from typing import Optional, Sequence, Type, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader

from domain.dist_match.config import DistMatchConfig
from domain.dist_match.modules.dist_match import DistMatch
from domain.dist_match.modules.model import Linear
from domain.seg_rnn.modules.model import SegRNN
from domain._common.trainers._base_torch_trainer import BaseTorchTrainer
from domain._common.losses.metrics_torch import mse, mae, kl_divergence, log_cosh_loss, uni_kl_divergence
from domain._common.losses.softdtw_loss.soft_dtw_torch import SoftDTW


class DistMatchTrainer(BaseTorchTrainer):
    def __init__(self, config: DistMatchConfig, device: torch.device):
        self.config: DistMatchConfig
        self.model: DistMatch
        # self.predictor: Linear = Linear(config.model).to(device)
        self.predictor: SegRNN = SegRNN(config.model).to(device)
        super(DistMatchTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[DistMatch]:
        return DistMatch

    def get_experiment_key(self, config: DistMatchConfig):
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
            torch.optim.Adam(self.predictor.parameters(), self.config.lr.init),
        )

    def _predict_stats(self, batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, y, *_ = batch
        y = y[:, -self.config.model.pred_len:, :]

        x_enc = self.model.encode(x)
        y_enc = self.model.encode(y)

        x_mean, x_std = x_enc.mean(-2), x_enc.std(-2)
        y_mean, y_std = y_enc.mean(-2), y_enc.std(-2)
        return (x_mean, x_std, y_mean, y_std), (x, y)

    def _predict_rec(self, batch) -> Tuple[Tuple[Tensor], Tuple[Tensor, Tensor]]:
        x, y, *_ = batch
        y = y[:, -self.config.model.pred_len:, :]

        x_rec = self.model.decode(self.model.encode(x))
        y_rec = self.model.decode(self.model.encode(y))

        return (x_rec, y_rec), (x, y)

    def _predict(self, batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, y, *_ = batch
        y = y[:, -self.config.model.pred_len:, :]

        preds = self.model.decode(self.predictor(self.model.encode(x)))

        return (preds), (x, y)

    def _compute_rec_loss(self, batch) -> Tuple[Tuple[Tensor], Tensor, Dict[str, Any]]:
        (x_rec, y_rec), (x, y) = self._predict_rec(batch)
        x_mae = mse(x_rec, x)
        y_mae = mse(y_rec, y)
        pred_loss = x_mae + y_mae

        return (pred_loss,), {
            "x_mse": x_mae.item(),
            "y_mse": y_mae.item(),
        }

    def _compute_pred_loss(self, batch) -> Tuple[Tuple[Tensor], Tensor, Dict[str, Any]]:
        (preds), (x, y) = self._predict(batch)
        loss = mse(preds, y)

        return (loss,), {
            "pred_loss": loss.item(),
        }

    def _compute_stat_loss(self, batch) -> Tuple[Tuple[Tensor], Tensor, Dict[str, Any]]:
        preds, (x, y) = self._predict_stats(batch)

        (x_mean, x_std, y_mean, y_std) = preds

        stat_loss = uni_kl_divergence(x_mean, x_std, y_mean, y_std)
        scale = 0.2
        regularization = (x_std.mean() + y_std.mean()) / 2
        stat_loss = stat_loss.mean() + regularization * scale

        metrics = {
            "Loss": stat_loss.item(),
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
        stat_optim, pred_optim = optimizers or (None, None)

        if stat_optim is not None:
            stat_optim.zero_grad()

        (stat_loss,), aux_data = self._compute_stat_loss(batch)

        if stat_optim is not None:
            stat_loss.backward()
            stat_optim.step()

        if stat_optim is not None:
            stat_optim.zero_grad()

        (rec_loss,), loss_data = self._compute_rec_loss(batch)

        if stat_optim is not None:
            rec_loss.backward()
            stat_optim.step()

        rec_loss = rec_loss.detach()
        aux_data.update(loss_data)
        loss = stat_loss

        if epoch >= self.config.lr.n_warmup_steps:
            if epoch == self.config.lr.n_warmup_steps:
                self.early_stopping.reset()

            if pred_optim is not None:
                pred_optim.zero_grad()
                stat_optim.zero_grad()

            (pred_loss,), loss_data = self._compute_pred_loss(batch)

            if pred_optim is not None:
                pred_loss.backward()
                pred_optim.step()
                stat_optim.step()

            pred_loss = pred_loss.detach()
            aux_data.update(loss_data)
            loss = pred_loss

        return (loss, ), aux_data

    def evaluate(self, data: DataLoader = None, stat_split: str = None, visualization_path: str = None, epoch: int = None):
        model_state = self.predictor.training
        self.predictor.eval()
        out = super().evaluate(data, stat_split, visualization_path, epoch)
        self.predictor.train(model_state)
        return out

    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with torch.no_grad():
            (x_rec, y_rec), (x, y), *_ = self._predict_rec(batch)
            preds = self._predict(batch)[0]

            x, y, x_rec, y_rec, preds = [
                t.detach()[0, :, -1].cpu()
                for t in [x, y, x_rec, y_rec, preds]
            ]

            gt = torch.concat([x, y])
            recs = torch.concat([x_rec, y_rec])
            preds = torch.concat([x, preds])

            input_len = len(x)

            # matplotlib.rcParams.update({'font.size': 15})
            plt.plot(preds, label='Predictions', linewidth=2)
            plt.plot(gt, label='Ground Truth', linewidth=2)
            plt.plot(recs, label='Recovered', linewidth=2)
            plt.vlines(input_len, -2, 0)

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
