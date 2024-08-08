from dataclasses import dataclass, field
from domain._common.data.dataset import SequenceSize, DATASET_LOADER_KEY_MAP
from generics import BaseConfig


@dataclass
class SSMConfig(BaseConfig):

    @dataclass
    class ModelConfig:
        n_channels: int = 7
        n_date_channels: int = 6
        seq_len: int = 336
        label_len: int = 0
        pred_len: int = 96
        d_model: int = 32
        d_inner: int = 64
        d_state: int = 16
        d_conv: int = 4
        d_dt: int = 32
        dt_min: int = None
        dt_max: int = None
        d_col: int = 32
        patch_size: int = 16
        n_blocks: int = 1
        dropout: float = 0.5

    @dataclass
    class DataConfig:
        @dataclass
        class DatasetConfig:
            path: str = 'data/ETTh1.csv'
            size: SequenceSize = field(init=False, repr=False)

        dataset: DatasetConfig = DatasetConfig()
        loader: str = 'change_point'
        batch_size: int = 8

    @dataclass
    class LearningRateConfig:
        rec: float = 1e-4
        pred: float = 1e-4
        decay: float = 0.8
        n_warmup_epochs: int = 3

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
