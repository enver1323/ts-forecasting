from domain._common.data.dataset import SequenceSize
from dataclasses import dataclass, asdict, field

@dataclass
class BaseConfig:
    seed: int = 1024

    @dataclass
    class DataConfig:
        @dataclass
        class DatasetConfig:
            path: str = 'data/ETTh1.csv'
            size: SequenceSize = field(init=False, repr=False)

        dataset: DatasetConfig = DatasetConfig()
        loader: str = 'common'
        batch_size: int = 32

    n_epochs: int = 100
    data: DataConfig = DataConfig()
    model: object = object()

    patience: int = 3

    def to_dict(self):
        return asdict(self)