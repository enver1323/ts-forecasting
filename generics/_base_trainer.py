from abc import ABC, abstractmethod
import torch
from common.wandb_log import HasLogging
from generics._base_config import BaseConfig
from typing import Type


class BaseTrainer(ABC):
    def __init__(self, config: BaseConfig, device: torch.device):
        self.config = config
        self.device = device

        is_logged = bool(self.config.wandb_log)
        self.log = HasLogging(is_logged)

    @property
    def experiment_key(self) -> str:
        return str(self.config)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, data=None):
        pass

    @abstractmethod
    def test(self, data=None):
        pass
