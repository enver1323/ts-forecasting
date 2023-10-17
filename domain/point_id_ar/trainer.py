from typing import Callable, Type

import torch

from domain.point_id import PointIDTrainer
from domain.point_id_ar.modules.model import PointIDAR, compute_loss
from domain.point_id_ar.config import PointIDARConfig
from generics import BaseConfig


class PointIDARTrainer(PointIDTrainer):
    def __init__(self, config: BaseConfig, device: torch.device):
        self.config: PointIDARConfig
        super(PointIDARTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[PointIDAR]:
        return PointIDAR

    @property
    def experiment_key(self):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = self.config.model.seq_len
        pred_len = self.config.model.pred_len
        n_choices = self.config.model.n_choices
        data = self.config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len})_choices_{n_choices}_WPointCompMLP_WOLastSub_WRevIN"
    
    @property
    def criterion(self) -> Callable:
        return compute_loss