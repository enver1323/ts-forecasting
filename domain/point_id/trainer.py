import os
from typing import Optional, Sequence, Callable, Type, Tuple, Dict, Any
from dataclasses import asdict

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from domain.point_id.modules.model import PointID
from domain.point_id.config import PointIDConfig
from domain._common.trainers._base_torch_trainer import BaseTorchTrainer
from domain._common.losses.softdtw_loss.soft_dtw_torch import SoftDTW
from domain._common.losses.metrics_torch import mse, mae
from generics import BaseConfig


class PointIDTrainer(BaseTorchTrainer):
    def __init__(self, config: BaseConfig, device: torch.device):
        self.config: PointIDConfig

        super(PointIDTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[PointID]:
        return PointID

    def get_experiment_key(self, config: PointIDConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        n_points = config.model.n_points
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len})_pts_{n_points}"

    def _get_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        return (
            torch.optim.AdamW(self.model.parameters(), self.config.lr.init),
        )

    def __compute_loss(self, x: Tensor, y: Tensor) -> Tuple[Tuple[Tensor], Dict[str, Any]]:
        y = y[:, -self.model.pred_len:, :]
        pred_y = self.model(x)

        # loss, soft_dtw_loss, path_dtw_loss = dilate_loss(pred_y, y, normalize=True)
        # loss = loss.mean()
        # soft_dtw_loss = soft_dtw_loss.mean()
        # path_dtw_loss = path_dtw_loss.mean()

        soft_dtw_loss = SoftDTW(normalize=False)(pred_y, y)
        soft_dtw_loss = soft_dtw_loss.mean()

        mse_loss = mse(pred_y, y)
        mse_scale = 100  # 200

        loss = soft_dtw_loss + mse_scale * mse_loss

        metrics = {
            "Loss": loss.item(),
            "SoftDTW": soft_dtw_loss.item(),
            # "PathDTW": path_dtw_loss.item(),
            "MSE": mse_loss.item(),
            "MAE": mae(pred_y, y).item(),
        }

        return (loss,), metrics

    def _step(self, batch: Sequence[torch.Tensor], optimizers: Optional[Sequence[torch.optim.Optimizer]] = None):
        optimizer, = optimizers or (None,)
        x, y, *_ = batch

        (loss,), aux_data = self.__compute_loss(x, y)

        if optimizer is not None:
            optimizer.zero_grad()

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        return (loss, ), aux_data

    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with torch.no_grad():
            x, y, *_ = batch
            x = x.float().to(self.device)

            pred = self.model(x).detach().cpu()
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

    def write_all_results(self):
        with open(self.RESULTS_PATH, 'a+') as file:
            file.write(self.experiment_key + '\n')
            self.log.write_text_all(file)
            file.write('\n\n')
