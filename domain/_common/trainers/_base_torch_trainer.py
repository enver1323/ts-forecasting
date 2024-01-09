import os
from abc import abstractmethod
from typing import Optional, Sequence, Type, Tuple, Dict, Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain._common.trainers.early_stopping.early_stopping_torch import EarlyStopping
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
    def _get_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        pass

    @abstractmethod
    def _step(self, batch: Sequence[torch.Tensor], optimizers: Optional[Sequence[torch.optim.Optimizer]] = None) -> Tuple[Sequence[Tensor], Dict[str, Any]]:
        pass

    def train(self):
        optimizers = self._get_optimizers()

        for epoch in range(self.config.n_epochs):
            self.log.reset_stat('train')
            p_bar = tqdm(self.train_data)
            total_loss = 0
            for step, batch in enumerate(p_bar):
                batch = [datum.float().to(self.device) for datum in batch]
                losses, aux_data = self._step(batch, optimizers)

                total_loss += losses[0].item()

                self.log.add_stat('train', aux_data)

                p_bar.set_description_str(
                    f"[Epoch {epoch}]: Train_loss: {total_loss / (step + 1):.3f}"
                )

            self.log.scale_stat('train', len(self.train_data))

            valid_loss = self.evaluate(self.valid_data, 'valid')
            test_loss = self.evaluate(self.test_data, 'test', os.path.join(
                self.PLOT_PATH, f'epoch_{epoch}'))
            print(
                f"[Epoch {epoch}]: Valid Loss: {valid_loss:.3f} | Test Loss: {test_loss:.3f}")

            if not self.early_stopping.step(valid_loss, self.model, f"checkpoints/{self.experiment_key}"):
                break

    def evaluate(self, data: Optional[DataLoader] = None, stat_split: Optional[str] = None, visualization_path: Optional[str] = None):
        with torch.no_grad():
            data = self.valid_data if data is None else data

            if stat_split is not None:
                self.log.reset_stat(stat_split)

            total_loss = 0

            for step, batch in enumerate(data):
                batch = [datum.float().to(self.device) for datum in batch]
                losses, aux_data = self._step(batch)
                total_loss += losses[0].item()
                if stat_split is not None:
                    self.log.add_stat(stat_split, aux_data)

                if visualization_path is not None:
                    self.visualize(batch, os.path.join(
                        visualization_path, f"plot_{step}.png"))

            if stat_split is not None:
                self.log.scale_stat(stat_split, len(data))

            return total_loss / len(data)

    def test(self):
        self.early_stopping.load_checkpoint(self.model, f"checkpoints/{self.experiment_key}")
        loss = self.evaluate(self.test_data, 'test',
                             os.path.join(self.PLOT_PATH, 'test'))

        self.write_all_results()

        return loss

    @abstractmethod
    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        pass

    def write_all_results(self):
        with open(self.RESULTS_PATH, 'a+') as file:
            file.write(self.experiment_key + '\n')
            self.log.write_text_all(file)
            file.write('\n\n')
