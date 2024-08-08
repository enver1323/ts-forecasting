import os
from abc import abstractmethod
from typing import Optional, Sequence, Type, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from domain._common.trainers.early_stopping.early_stopping_torch import EarlyStopping
from domain._common.losses.metrics_torch import mse, mae
from generics import BaseTrainer, BaseConfig


class BaseTorchTrainer(BaseTrainer):
    def __init__(self, config: BaseConfig, device: torch.device):
        super(BaseTorchTrainer, self).__init__(config)
        self.device = device

        self.model: nn.Module = self.model_type(config.model).to(self.device)
        self.PLOT_PATH = f'plots/{self.experiment_key}/'
        self.RESULTS_PATH = 'results.txt'
        self._set_data(self.config.data)

        self.early_stopping = EarlyStopping(self.config.patience)

    @property
    @abstractmethod
    def model_type(self) -> Type[nn.Module]:
        pass

    @abstractmethod
    def get_experiment_key(self, config: BaseConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]

        return f"{model_name}_{config}"

    @property
    def experiment_key(self):
        return self.get_experiment_key(self.config)

    @abstractmethod
    def _get_optimizers(self) -> Sequence[Optimizer]:
        pass

    def _get_schedulers(self, optimizers: Sequence[Optimizer]) -> Sequence[Optimizer]:
        return []

    def _scheduler_step(
        self,
        schedulers: Sequence[LRScheduler],
        optimizers: Sequence[Optimizer],
        loss: torch.Tensor,
        epoch: int
    ):
        pass

    @abstractmethod
    def _predict(self, batch) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @abstractmethod
    def _step(self, batch: Sequence[torch.Tensor], optimizers: Optional[Sequence[Optimizer]] = None, epoch: Optional[int] = None) -> Tuple[Sequence[Tensor], Dict[str, Any]]:
        pass

    def train(self):
        model_state = self.model.training
        self.model.train()

        optimizers = self._get_optimizers()
        schedulers = self._get_schedulers(optimizers)

        for epoch in range(self.config.n_epochs):
            self.log.reset_stat('train')
            total_loss = 0
            for batch in self.train_data:
                batch = [datum.float().to(self.device) for datum in batch]
                losses, aux_data = self._step(batch, optimizers, epoch)

                total_loss += losses[0].item()

                self.log.add_stat('train', aux_data)

            self.log.scale_stat('train', len(self.train_data))
            self.log.show_stat('train')

            valid_loss = self.evaluate(self.valid_data, 'valid', epoch=epoch)
            test_loss = self.evaluate(self.test_data, 'test', epoch=epoch)

            print(
                f"[Epoch {epoch}]: Valid Loss: {valid_loss:.3f} | Test Loss: {test_loss:.3f}")

            self._scheduler_step(schedulers, optimizers, valid_loss, epoch)

            if not self.early_stopping.step(valid_loss, self.model, f"checkpoints/{self.experiment_key}"):
                break

        self.model.train(model_state)

    def evaluate(self, data: Optional[DataLoader] = None, stat_split: Optional[str] = None, visualization_path: Optional[str] = None, epoch: Optional[int] = None):
        model_state = self.model.training
        self.model.eval()
        with torch.no_grad():
            data = self.valid_data if data is None else data

            if stat_split is not None:
                self.log.reset_stat(stat_split)

            total_loss = 0

            for step, batch in enumerate(data):
                batch = [datum.float().to(self.device) for datum in batch]
                losses, aux_data = self._step(batch, epoch=epoch)
                total_loss += losses[0].item()
                if stat_split is not None:
                    self.log.add_stat(stat_split, aux_data)

                if visualization_path is not None:
                    self.visualize(batch, os.path.join(
                        visualization_path, f"plot_{step}.png"))

            if stat_split is not None:
                self.log.scale_stat(stat_split, len(data))
                self.log.show_stat(stat_split)

        self.model.train(model_state)
        return total_loss / len(data)

    def test(self):
        self.early_stopping.load_checkpoint(
            self.model, f"checkpoints/{self.experiment_key}")
        test_loss = self.evaluate(self.test_data, 'test', self.PLOT_PATH)
        self.write_all_results()

        return test_loss

    @abstractmethod
    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        pass

    def write_all_results(self):
        with open(self.RESULTS_PATH, 'a+') as file:
            file.write(self.experiment_key + '\n')
            self.log.write_text_all(file)
            file.write('\n\n')
