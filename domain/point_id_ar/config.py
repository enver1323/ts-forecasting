from dataclasses import dataclass, field
from domain._common.data.dataset import SequenceSize, DATASET_LOADER_KEY_MAP
from generics import BaseConfig


@dataclass
class PointIDARConfig(BaseConfig):

    @dataclass
    class ModelConfig:
        n_channels: int = 7
        seq_len: int = 96
        label_len: int = 48
        pred_len: int = 96
        window_len: int = 96
        kernel_size: int = 25
        n_choices: int = 4

    @dataclass
    class DataConfig:
        @dataclass
        class DatasetConfig:
            path: str = 'data/ETTh1.csv'
            size: SequenceSize = field(init=False, repr=False)

        dataset: DatasetConfig = DatasetConfig()
        loader: str = 'common'
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
        self.data.dataset.size = SequenceSize(
            seq_len=self.model.seq_len,
            label_len=self.model.label_len,
            pred_len=self.model.pred_len
        )
        assert self.data.loader in DATASET_LOADER_KEY_MAP
