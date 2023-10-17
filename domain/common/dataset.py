import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Union, NamedTuple
from generics import DataSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
from enum import Enum
from typing import Type, Dict


class SequenceSize(NamedTuple):
    source_len: int
    overlap_len: int
    pred_len: int


class DataSplitSize(NamedTuple):
    train_len: Union[float, int]
    valid_len: Optional[Union[float, int]]
    test_len: Union[float, int]


class CommonTSDataset(Dataset):
    def __init__(
            self,
            path: str,
            size: SequenceSize,
            data_split: DataSplit,
            split_sizes: DataSplitSize = DataSplitSize(0.7, None, 0.2),
            scale: bool = True,
    ):
        super(CommonTSDataset, self).__init__()

        self.source_len, self.overlap_len, self.pred_len = size

        self.size = size
        self.data_split = data_split
        self.split_sizes = split_sizes
        self.scale = scale
        self.scaler: StandardScaler

        self.__read_data__(path)

    def _get_data_split(self, data_len: int):
        train_len, test_len = (
            int(item * data_len) if isinstance(item, float) else item
            for item in (self.split_sizes.train_len, self.split_sizes.test_len)
        )
        valid_len = self.split_sizes.valid_len
        if valid_len is None:
            valid_len = data_len - train_len - test_len
        elif isinstance(valid_len, float):
            valid_len = int(valid_len)

        return {
            DataSplit.train: slice(0, train_len),
            DataSplit.valid: slice(train_len - self.source_len, train_len + valid_len),
            DataSplit.test: slice(train_len + valid_len - self.source_len, train_len + valid_len + test_len),
        }

    def _scale_data(self, train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
        self.scaler = StandardScaler()
        self.scaler.fit(train_df)
        return self.scaler.transform(target_df)

    def _get_date_df(self, df_series: pd.Series):
        df_stamp = pd.DataFrame()
        df_stamp['date'] = pd.to_datetime(df_series)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
        df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)

        return df_stamp.drop(columns=['date']).values

    def __read_data__(self, path: str):
        src_df = pd.read_csv(path)
        split_slices = self._get_data_split(len(src_df))
        df = src_df[split_slices[self.data_split]]

        date_df = self._get_date_df(df['date'])
        df = df.drop(columns=['date'])

        if self.scale:
            df = self._scale_data(
                src_df.loc[split_slices[DataSplit.train], df.columns],
                df
            )
        else:
            df = df.values

        self.data = df
        self.date_data = date_df

    def __getitem__(self, index):
        source_slice = slice(index, index + self.source_len)
        target_slice = slice(source_slice.stop - self.overlap_len,
                             source_slice.stop + self.pred_len)
        return (
            self.data[source_slice],
            self.data[target_slice],
            self.date_data[source_slice],
            self.date_data[target_slice],
        )

    def __len__(self):
        entry_len = self.source_len + self.pred_len
        return len(self.data) - entry_len + 1

    def inverse_scale(self, data):
        if not self.scale:
            raise ModuleNotFoundError("Scale was not found")

        return self.scaler.inverse_transform(data)


class ETTHDataset(CommonTSDataset):
    def __init__(
        self,
        path: str,
        size: SequenceSize,
        data_split: DataSplit,
        split_sizes: Optional[DataSplitSize] = None,
        scale: bool = True,
    ):
        if split_sizes is None:
            month = 30 * 24
            split_sizes = DataSplitSize(
                train_len=12 * month,
                valid_len=4 * month,
                test_len=4 * month
            )

        super(ETTHDataset, self).__init__(
            path=path,
            size=size,
            data_split=data_split,
            split_sizes=split_sizes,
            scale=scale,
        )


class ETTMDataset(CommonTSDataset):
    def __init__(
        self,
        path: str,
        size: SequenceSize,
        data_split: DataSplit,
        split_sizes: Optional[DataSplitSize] = None,
        scale: bool = True,
    ):
        if split_sizes is None:
            month = 30 * 24 * 4
            split_sizes = DataSplitSize(
                train_len=12 * month,
                valid_len=4 * month,
                test_len=4 * month
            )

        super(ETTMDataset, self).__init__(
            path=path,
            size=size,
            data_split=data_split,
            split_sizes=split_sizes,
            scale=scale,
        )


class DatasetLoaders(Enum):
    etth = ('etth', ETTHDataset)
    ettm = ('ettm', ETTMDataset)
    common = ('common', CommonTSDataset)

    def __init__(self, key: str, config: Type[CommonTSDataset]):
        self.key = key
        self.config = config


DATASET_LOADER_KEY_MAP: Dict[str, Type[CommonTSDataset]] = {
    loader.key: loader.config for loader in DatasetLoaders
}
