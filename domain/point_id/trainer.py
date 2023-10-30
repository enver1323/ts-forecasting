import os
from typing import Optional, Sequence, Callable, Type
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from domain.point_id.modules.model import PointID, compute_loss
from domain.point_id.config import PointIDConfig
from domain.common.utils import EarlyStopping
from domain.common.dataset import DATASET_LOADER_KEY_MAP, CommonTSDataset
from generics import BaseTrainer, DataSplit, BaseConfig


class PointIDTrainer(BaseTrainer):
    def __init__(self, config: BaseConfig, device: torch.device):
        self.config: PointIDConfig
        super(PointIDTrainer, self).__init__(config, device)

        self.model: PointID = self.model_type(
            config=self.config.model).to(self.device)
        self.PLOT_PATH = f'plots/{self.experiment_key}/'
        self.RESULTS_PATH = 'results.txt'
        self.__set_data(self.config.data)

        self.early_stopping = EarlyStopping(self.config.patience)

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

    @property
    def experiment_key(self):
        return self.get_experiment_key(self.config)

    @property
    def criterion(self) -> Callable:
        return compute_loss

    def __get_data_loader(self, config: PointIDConfig.DataConfig.DatasetConfig, data_loader: Type[CommonTSDataset], data_split: DataSplit, **kwargs) -> DataLoader:
        dataset = data_loader(**{**(asdict(config)), "data_split": data_split})
        return DataLoader(dataset, **kwargs)

    def __set_data(self, config: PointIDConfig.DataConfig):
        dataset_config = config.dataset
        data_loader = DATASET_LOADER_KEY_MAP[config.loader]
        self.train_data = self.__get_data_loader(
            dataset_config, data_loader, DataSplit.train, batch_size=config.batch_size, shuffle=True
        )
        self.valid_data = self.__get_data_loader(
            dataset_config, data_loader, DataSplit.valid, batch_size=config.batch_size, shuffle=True
        )
        self.test_data = self.__get_data_loader(
            dataset_config, data_loader, DataSplit.test, batch_size=config.batch_size
        )

    def train(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), self.config.lr.init
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5], gamma=self.config.lr.gamma)

        for epoch in range(self.config.n_epochs):
            self.log.reset_stat('train')
            p_bar = tqdm(self.train_data)
            total_loss = 0
            for step, batch in enumerate(p_bar):
                batch = [datum.float().to(self.device) for datum in batch]

                optimizer.zero_grad()

                loss, aux_data = self.criterion(self.model, *batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

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
                loss, aux_data = self.criterion(self.model, *batch)
                total_loss += loss.item()
                if stat_split is not None:
                    self.log.add_stat(stat_split, aux_data)

                if visualization_path is not None:
                    self.visualize(batch, os.path.join(
                        visualization_path, f"plot_{step}.png"))

            if stat_split is not None:
                self.log.scale_stat(stat_split, len(data))

            return total_loss / len(data)

    def test(self):
        self.early_stopping.load_checkpoint(
            self.model, f"checkpoints/{self.experiment_key}")

        loss = self.evaluate(self.test_data, 'test',
                             os.path.join(self.PLOT_PATH, 'test'))

        self.log.wandb_log_all()
        self.write_all_results()

        return loss

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

            # y = torch.concat([x[:, -1:, :], y], dim=-2)
            # pred = torch.concat([x[:, -1:, :], pred], dim=-2)
            y = torch.concat([x, y], dim=-2)
            pred = torch.concat([x, pred], dim=-2)

            seq_len = self.config.model.seq_len
            pred_len = self.config.model.pred_len

            # x_ticks = torch.arange(0, seq_len)
            # y_ticks = torch.arange(seq_len - 1, seq_len + pred_len)

            # plt.plot(x_ticks, x[0, :, -1])
            # plt.plot(y_ticks, y[0, :, -1])
            # plt.plot(y_ticks, pred[0, :, -1])
            plt.plot(y[0, :, -1], label='GroundTruth', linewidth=2)
            plt.plot(pred[0, :, -1], label='Prediction', linewidth=2)

            plt.tight_layout()
            plt.legend()
            plt.savefig(filepath)
            plt.close()

    def write_all_results(self):
        with open(self.RESULTS_PATH, 'a+') as file:
            file.write(self.experiment_key + '\n')
            self.log.write_text_all(file)
            file.write('\n\n')
