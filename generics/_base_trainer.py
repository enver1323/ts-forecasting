from abc import ABC, abstractmethod
from typing import Type
from dataclasses import asdict

from torch.utils.data import DataLoader

from generics._base_config import BaseConfig
from common.logging import HasLogging
from domain._common.data.dataset import DATASET_LOADER_KEY_MAP, CommonTSDataset, DataSplit


class BaseTrainer(ABC):
    def __init__(self, config: BaseConfig):
        self.config = config

        self.log = HasLogging()

    @property
    def experiment_key(self) -> str:
        return str(self.config)

    def _get_data_loader(
        self,
        config: BaseConfig.DataConfig.DatasetConfig,
        data_loader: Type[CommonTSDataset],
        data_split: DataSplit,
        **kwargs
    ) -> DataLoader:
        dataset = data_loader(**{**(asdict(config)), "data_split": data_split})
        return DataLoader(dataset, **kwargs)

    def _set_data(self, config: BaseConfig.DataConfig):
        dataset_config = config.dataset
        data_loader = DATASET_LOADER_KEY_MAP[config.loader]
        self.train_data = self._get_data_loader(
            dataset_config, data_loader, DataSplit.train, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        self.valid_data = self._get_data_loader(
            dataset_config, data_loader, DataSplit.valid, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        self.test_data = self._get_data_loader(
            dataset_config, data_loader, DataSplit.test, batch_size=config.batch_size, drop_last=True
        )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, data=None):
        pass

    @abstractmethod
    def test(self, data=None):
        pass
