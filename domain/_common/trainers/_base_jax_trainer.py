from abc import abstractmethod
from dataclasses import asdict
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Type

import numpy as np
from jaxtyping import Array
import jax
import jax.random as jrandom
from jax.random import KeyArray
import equinox as eqx
import optax
from torch.utils.data import DataLoader

from domain._common.trainers.early_stopping.early_stopping_jax import EarlyStopping
from domain._common.data.np_loader import NumpyLoader
from domain._common.data.dataset import DataSplit, CommonTSDataset
from generics._base_trainer import BaseTrainer
from generics._base_config import BaseConfig


class BaseJaxTrainer(BaseTrainer):
    def __init__(self, config: BaseConfig, key: KeyArray):
        super(BaseJaxTrainer, self).__init__(config)
        self.key, self.train_key = jrandom.split(key, 2)

        self.model: eqx.Module = self.model_type(
            self.config.model, key=self.key)
        self.PLOT_PATH = f'plots/{self.experiment_key}/'
        self.RESULTS_PATH = 'results.txt'
        self._set_data(self.config.data)

        self.early_stopping = EarlyStopping(self.config.patience)

    @property
    @abstractmethod
    def model_type(self) -> Type[eqx.Module]:
        pass

    def get_experiment_key(self, config: BaseConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]

        return f"{model_name}_{str(config)}"

    @property
    def experiment_key(self):
        return self.get_experiment_key(self.config)

    def _get_data_loader(
        self,
        config: BaseConfig.DataConfig.DatasetConfig,
        data_loader: Type[CommonTSDataset],
        data_split: DataSplit,
        **kwargs
    ) -> NumpyLoader:
        dataset = data_loader(**{**(asdict(config)), "data_split": data_split})
        return NumpyLoader(dataset, **kwargs)

    @abstractmethod
    def _init_optimizers_w_states(self) -> Sequence[Tuple[optax.GradientTransformation, optax.OptState]]:
        pass

    @abstractmethod
    def _step(
        self,
        model: eqx.Module,
        batch: Sequence[Array],
        optimizers: Optional[Sequence[
            Tuple[optax.GradientTransformation, optax.OptState]
        ]] = None,
        *,
        key: Optional[KeyArray] = None
    ) -> Tuple[Tuple[Array, ...], Dict[str, Any]]:
        pass

    def train(self):
        optimizers_w_states = self._init_optimizers_w_states()

        for epoch in range(self.config.n_epochs):
            self.log.reset_stat('train')
            total_loss = 0
            for batch in self.train_data:
                _, self.train_key = jrandom.split(self.train_key, 2)
                self.model, loss, aux_data = self._step(
                    self.model, batch, optimizers_w_states, key=self.train_key)

                total_loss += np.asarray(loss)

                self.log.add_stat('train', aux_data)

            self.log.scale_stat('train', len(self.train_data))
            self.log.show_stat('train')

            # test_loss = self.evaluate(self.test_data, 'test', os.path.join(
            #     self.PLOT_PATH, f'epoch_{epoch}'))
            valid_loss = self.evaluate(self.valid_data, 'valid')
            test_loss = self.evaluate(self.test_data, 'test')

            print(
                f"[Epoch {epoch}]: Valid Loss: {valid_loss:.3f} | Test Loss: {test_loss:.3f}")

            if not self.early_stopping.step(valid_loss, self.model, f"checkpoints/{self.experiment_key}"):
                break

    def evaluate(self, data: Optional[DataLoader] = None, stat_split: Optional[str] = None, visualization_path: Optional[str] = None):
        self.model = eqx.nn.inference_mode(self.model)
        data = self.valid_data if data is None else data

        if stat_split is not None:
            self.log.reset_stat(stat_split)

        total_loss = 0

        for step, batch in enumerate(data):
            batch = [jax.lax.stop_gradient(datum) for datum in batch]
            _, loss, aux_data = self._step(self.model, batch)
            total_loss += loss.item()
            if stat_split is not None:
                self.log.add_stat(stat_split, aux_data)

            if visualization_path is not None:
                self.visualize(batch, os.path.join(
                    visualization_path, f"plot_{step}.png"))

        self.model = eqx.nn.inference_mode(self.model, value=False)

        if stat_split is not None:
            self.log.scale_stat(stat_split, len(data))
            self.log.show_stat(stat_split)

        return total_loss / len(data)

    def test(self):
        self.model = self.early_stopping.load_checkpoint(
            self.model, f"checkpoints/{self.experiment_key}")

        loss = self.evaluate(
            self.test_data, 'test', os.path.join(self.PLOT_PATH, 'test')
        )
        # loss = self.evaluate(self.test_data, 'test')

        self.write_all_results()

        return loss

    @abstractmethod
    def visualize(self, batch: Sequence[Array], filepath: str):
        pass

    def write_all_results(self):
        with open(self.RESULTS_PATH, 'a+') as file:
            file.write(self.experiment_key + '\n')
            self.log.write_text_all(file)
            file.write('\n\n')
