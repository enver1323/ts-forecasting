import os
from typing import Optional, Sequence, Type, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

from domain.keeper.modules.model import Keeper
from domain.keeper.config import KeeperConfig
from domain._common.trainers._base_torch_trainer import BaseTorchTrainer
from domain._common.losses.metrics_torch import mse, mae
from domain._common.losses.softdtw_loss.soft_dtw_torch import SoftDTW


class KeeperTrainer(BaseTorchTrainer):
    def __init__(self, config: KeeperConfig, device: torch.device):
        self.config: KeeperConfig
        self.model: Keeper

        super(KeeperTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[Keeper]:
        return Keeper

    def get_experiment_key(self, config: KeeperConfig):
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
        pred_len = self.config.data.dataset.size.pred_len
        y = y[:, -pred_len:, :]

        preds = self.model(x)
        preds = preds[:, -pred_len:, :]

        return preds, (x, y)

    def _compute_loss(self, batch) -> Tuple[Tuple[Tensor], Tensor, Dict[str, Any]]:
        (preds, *_), (x, y) = self._predict(batch)
        pred_mae = mae(preds, y)
        pred_mse = mse(preds, y)
        pred_loss = pred_mae + pred_mse

        return (pred_loss,), {
            "pred_mse": pred_mse.item(),
            "pred_mae": pred_mae.item(),
        }

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
        optim, = optimizers or (None,)

        if optim is not None:
            optim.zero_grad()

        (loss,), aux_data = self._compute_loss(batch)

        if optim is not None:
            loss.backward()
            optim.step()
        loss = loss.detach()

        return (loss, ), aux_data

    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with torch.no_grad():
            preds, (x, y), *_ = self._predict(batch)
            x = x.detach().cpu()[0, :, -1]
            y = y.detach().cpu()[0, :, -1]
            preds = preds.detach().cpu()[0, :, -1]

            gt = torch.concat([x, y])
            preds = torch.concat([x, preds])

            input_len = len(x)

            # matplotlib.rcParams.update({'font.size': 15})
            plt.plot(preds, label='Predictions', linewidth=2)
            plt.plot(gt, label='Ground Truth', linewidth=2)
            plt.vlines(input_len, -2, 0)

            plt.tight_layout()
            plt.legend()
            plt.savefig(filepath)
            plt.close()
