from typing import Callable, Sequence, Type, Optional
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain.point_id_ar import PointIDARTrainer
from domain.lin_adapt.modules.model import LinAdapt, compute_loss
from domain.lin_adapt.config import LinAdaptConfig
from generics import BaseConfig


class LinAdaptTrainer(PointIDARTrainer):
    def __init__(self, config: BaseConfig, device: torch.device):
        self.config: LinAdaptConfig
        super(LinAdaptTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[LinAdapt]:
        return LinAdapt

    def get_experiment_key(self, config: LinAdaptConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        seq_len = config.model.seq_len
        pred_len = config.model.pred_len
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_({seq_len}->{pred_len})_SepLoss"
    
    @property
    def criterion(self) -> Callable:
        return compute_loss
    
    def _get_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        return (
            torch.optim.AdamW(self.model.parameters(), self.config.lr.init),
            torch.optim.AdamW(self.model.parameters(), self.config.lr.init),
            torch.optim.AdamW(self.model.parameters(), self.config.lr.init),
        )