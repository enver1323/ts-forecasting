from typing import Callable, Type, Optional
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
    
    def train(self):
        mse_optimizer = torch.optim.AdamW(
            self.model.parameters(), self.config.lr.init
        )
        dtw_optimizer = torch.optim.AdamW(
            self.model.parameters(), self.config.lr.init
        )
        norm_optimizer = torch.optim.AdamW(
            self.model.parameters(), self.config.lr.init
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5], gamma=self.config.lr.gamma)

        for epoch in range(self.config.n_epochs):
            self.log.reset_stat('train')
            p_bar = tqdm(self.train_data)
            total_loss = 0
            for step, batch in enumerate(p_bar):
                batch = [datum.float().to(self.device) for datum in batch]

                mse_optimizer.zero_grad()
                # dtw_optimizer.zero_grad()
                # norm_optimizer.zero_grad()

                (mse_loss, dtw_loss, norm_loss), aux_data = self.criterion(self.model, *batch)
                mse_loss.backward()
                mse_optimizer.step()

                # dtw_loss.backward()
                # dtw_optimizer.step()

                # norm_loss.backward()
                # norm_optimizer.step()

                total_loss += mse_loss.item()

                self.log.add_stat('train', aux_data)

                p_bar.set_description_str(
                    f"[Epoch {epoch}]: Train_loss: {total_loss / (step + 1):.3f}"
                )

            # scheduler.step()

            self.log.scale_stat('train', len(self.train_data))

            valid_loss = self.evaluate(self.valid_data, 'valid')
            test_loss = self.evaluate(self.test_data, 'test', os.path.join(
                self.PLOT_PATH, f'epoch_{epoch}'))
            self.log.wandb_log_all()
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
                (mse_loss, *_), aux_data = self.criterion(self.model, *batch)
                total_loss += mse_loss.item()
                if stat_split is not None:
                    self.log.add_stat(stat_split, aux_data)

                if visualization_path is not None:
                    self.visualize(batch, os.path.join(
                        visualization_path, f"plot_{step}.png"))

            if stat_split is not None:
                self.log.scale_stat(stat_split, len(data))

            return total_loss / len(data)