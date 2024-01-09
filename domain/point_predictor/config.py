from dataclasses import dataclass, field
from domain._common.data.dataset import DATASET_LOADER_KEY_MAP
from generics import BaseConfig


@dataclass
class PointPredictorConfig(BaseConfig):

    @dataclass
    class ModelConfig:
        n_channels: int = 7
        context_size: int = 720
        initial_size: int = 96
        window_size: int = 96
        hidden_size: int = 48
        kernel_size: int = 25

    @dataclass
    class DataConfig:
        @dataclass
        class DatasetConfig:
            path: str = 'data/ETTh1.csv'
            context_size: int = field(init=False, repr=False)

        dataset: DatasetConfig = DatasetConfig()
        loader: str = 'context'
        batch_size: int = 32

    @dataclass
    class LearningRateConfig:
        init: float = 1e-4
        gamma: float = 0.3

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()

    lr: LearningRateConfig = LearningRateConfig()

    n_epochs: int = 100
    patience: int = 2

    def __post_init__(self):
        self.data.dataset.context_size = self.model.context_size
        assert self.data.loader in DATASET_LOADER_KEY_MAP