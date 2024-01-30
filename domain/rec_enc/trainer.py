import os
from typing import Optional, Sequence, Type, Tuple, Dict, Any

import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

from domain.rec_enc.modules.model import RecEnc
from domain.rec_enc.config import RecEncConfig
from domain._common.trainers._base_torch_trainer import BaseTorchTrainer
from domain._common.losses.metrics_torch import mse, mae


class RecEncTrainer(BaseTorchTrainer):
    def __init__(self, config: RecEncConfig, device: torch.device):
        self.config: RecEncConfig

        super(RecEncTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[RecEnc]:
        return RecEnc

    def get_experiment_key(self, config: RecEncConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        patch_len = config.model.patch_len
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len}|{patch_len})"

    def _get_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        return (
            torch.optim.AdamW(self.model.parameters(), self.config.lr.init),
        )

    def _predict(self, batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        x, y, *_ = batch
        x = x.float().to(self.device)
        y = y.float()
        preds = self.model(x)
        return preds, (x, y)

    def _compute_loss(self, batch) -> Tuple[Tuple[Tensor], Dict[str, Any]]:
        preds, (_, y) = self._predict(batch)
        y = y[:, -self.config.model.pred_len:, :]

        loss = 0.5 * F.mse_loss(preds, y) + 0.5 * F.l1_loss(preds, y)
        # loss = F.l1_loss(pred_y, y)

        mse_loss = mse(preds.detach(), y)
        mae_loss = mae(preds.detach(), y)

        metrics = {
            "Loss": loss.item(),
            "mse": mse_loss.item(),
            "mae": mae_loss.item(),
        }

        return (loss,), metrics

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
            lr = lr * (decay ** ((epoch - 1) // 1))

        for optim in optimizers:
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def _step(self, batch: Sequence[torch.Tensor], optimizers: Optional[Sequence[torch.optim.Optimizer]] = None):
        optimizer, = optimizers or (None,)

        (loss,), aux_data = self._compute_loss(batch)

        if optimizer is not None:
            optimizer.zero_grad()

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        return (loss.detach(), ), aux_data

    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with torch.no_grad():
            pred, (x, y) = self._predict(batch)
            pred = pred.detach().cpu()
            x = x.detach().cpu()
            y = y.detach().cpu()

            y = y[:, -self.model.pred_len:, :]

            y = torch.concat([x, y], dim=-2)
            pred = torch.concat([x, pred], dim=-2)

            matplotlib.rcParams.update({'font.size': 15})
            plt.plot(pred[0, :, -1], label='Prediction', linewidth=2)
            plt.plot(y[0, :, -1], label='GroundTruth', linewidth=2)

            plt.tight_layout()
            plt.legend()
            plt.savefig(filepath)
            plt.close()
