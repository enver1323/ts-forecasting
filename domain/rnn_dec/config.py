from dataclasses import dataclass, field
from domain._common.data.dataset import SequenceSize, DATASET_LOADER_KEY_MAP
from generics import BaseConfig


@dataclass
class RNNDecConfig(BaseConfig):

    @dataclass
    class ModelConfig:
        n_channels: int = 7
        n_date_channels: int = 6
        seq_len: int = 720
        label_len: int = 0
        pred_len: int = 96
        patch_len: int = 48
        dropout: float = 0.5
        d_model: int = 512

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
        init: float = 1e-3
        decay: float = 0.8

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()

    lr: LearningRateConfig = LearningRateConfig()

    n_epochs: int = 30
    n_stat_epochs: int = 10
    patience: int = 10

    def __post_init__(self):
        self.data.dataset.size = SequenceSize(
            seq_len=self.model.seq_len,
            label_len=self.model.label_len,
            pred_len=self.model.pred_len
        )
        assert self.data.loader in DATASET_LOADER_KEY_MAP
