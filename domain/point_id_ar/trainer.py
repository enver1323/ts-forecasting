from typing import Sequence, Type, Optional

import torch

from domain.point_id.trainer import PointIDTrainer
from domain.point_id_ar.modules.model import PointIDAR
from domain.point_id_ar.config import PointIDARConfig
from domain._common.losses.metrics_torch import mse, mae
from domain._common.losses.softdtw_loss.soft_dtw_torch import SoftDTW
from generics import BaseConfig


class PointIDARTrainer(PointIDTrainer):
    def __init__(self, config: BaseConfig, device: torch.device):
        self.config: PointIDARConfig
        self.model: PointIDAR

        super(PointIDARTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[PointIDAR]:
        return PointIDAR

    def get_experiment_key(self, config: PointIDARConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        n_choices = config.model.n_choices
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len})_choices_{n_choices}_SepLossResScaled_GlobalRevIN_PointMLP"

    def _get_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        return (
            torch.optim.AdamW(
                self.model.parameters(), self.config.lr.init
            ), torch.optim.AdamW(
                self.model.parameters(), self.config.lr.init
            )
        )

    def _step(self, batch: Sequence[torch.Tensor], optimizers: Optional[Sequence[torch.optim.Optimizer]] = None):
        mse_optim, soft_dtw_optim = optimizers or (None, None)
        x, y, *_ = batch
        y = y[:, -self.model.pred_len:, :]

        if soft_dtw_optim is not None:
            soft_dtw_optim.zero_grad()
        pred_y = self.model(x)

        soft_dtw_loss = SoftDTW(normalize=False)(pred_y, y)
        soft_dtw_loss = soft_dtw_loss.mean()

        if soft_dtw_optim is not None:
            soft_dtw_loss.backward()
            soft_dtw_optim.step()

        if mse_optim is not None:
            mse_optim.zero_grad()

        pred_y = self.model(x)

        mse_loss = mse(pred_y, y)
        if mse_optim is not None:
            mse_loss.backward()
            mse_optim.step()

        metrics = {
            # "Loss": loss.item(),
            "SoftDTW": soft_dtw_loss.item(),
            # "PathDTW": path_dtw_loss.item(),
            "MSE": mse_loss.item(),
            "MAE": mae(pred_y, y).item(),
        }

        return (mse_loss, soft_dtw_loss), metrics
