import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Union, NamedTuple
from generics import DataSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
from enum import Enum
from typing import Type, Dict, Callable, Tuple, List


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
            split_sizes: Optional[DataSplitSize] = None,
            scale: bool = True,
    ):
        super(CommonTSDataset, self).__init__()

        self.source_len, self.overlap_len, self.pred_len = size

        self.size = size
        self.data_split = data_split
        self._original_data_len = None
        self.split_sizes = DataSplitSize(
            0.7, None, 0.2) if split_sizes is None else split_sizes
        self.scale = scale
        self.scaler: StandardScaler

        self.__read_data__(path)

    def _get_data_splits(self):
        data_len = self._original_data_len
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

        df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
        df_stamp['year'] = df_stamp.year - df_stamp.year.min()
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
        # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)

        return df_stamp.drop(columns=['date']).values

    def __read_data__(self, path: str):
        src_df = pd.read_csv(path)
        self._original_data_len = len(src_df)
        split_slices = self._get_data_splits()
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

    def __getitem__(self, idx):
        source_slice = slice(idx, idx + self.source_len)
        target_slice = slice(source_slice.stop - self.overlap_len,
                             source_slice.stop + self.pred_len)
        return (
            self.data[source_slice],
            self.data[target_slice],
            self.date_data[source_slice],
            self.date_data[target_slice],
            idx
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


class ContextDataset(CommonTSDataset):
    def __init__(
        self,
        path: str,
        context_size: int,
        data_split: DataSplit,
        split_sizes: Optional[DataSplitSize] = None,
        scale: bool = True,
    ):
        super(ContextDataset, self).__init__(path=path, size=SequenceSize(
            context_size, 0, 0), data_split=data_split, split_sizes=split_sizes, scale=scale)

    def __gettitem__(self, idx):
        context_slice = slice(idx, idx + self.source_len)

        return (
            self.data[context_slice],
            self.date_data[context_slice],
        )


class ChangePointDataset(CommonTSDataset):
    def __init__(
        self,
        path: str,
        size: SequenceSize,
        data_split: DataSplit,
        split_sizes: Optional[DataSplitSize] = None,
        scale: bool = True
    ):
        super(ChangePointDataset, self).__init__(
            path=path,
            size=size,
            data_split=data_split,
            split_sizes=split_sizes,
            scale=scale,
        )

    def _separate_cp_cols(
        self,
        df: pd.DataFrame,
        cp_col_id: Callable[[str], bool] = lambda col: col[-3:] == '_cp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        df_cols = list(df.columns)
        cp_cols = list(filter(cp_col_id, df_cols))
        data_cols = list(filter(lambda col: not (cp_col_id(col)), df_cols))

        cp_df = df[cp_cols]
        data_df = df[data_cols]
        return data_df, cp_df, data_cols, cp_cols

    def __read_data__(self, path: str):
        src_df = pd.read_csv(path)
        self._original_data_len = len(src_df)
        split_slices = self._get_data_splits()
        df = src_df[split_slices[self.data_split]]

        date_df = self._get_date_df(df['date'])
        df = df.drop(columns=['date'])

        data_df, cp_df, data_cols, cp_cols = self._separate_cp_cols(df)

        if self.scale:
            df = self._scale_data(
                src_df.loc[split_slices[DataSplit.train], data_cols],
                data_df
            )
        else:
            df = df.values

        self.data = df
        self.date_data = date_df
        self.cp_data = cp_df.values

    def __getitem__(self, idx):
        source_slice = slice(idx, idx + self.source_len)
        target_slice = slice(source_slice.stop - self.overlap_len,
                             source_slice.stop + self.pred_len)
        return (
            self.data[source_slice],
            self.data[target_slice],
            self.cp_data[source_slice],
            self.cp_data[target_slice],
            self.date_data[source_slice],
            self.date_data[target_slice],
            idx
        )


class PatchAlignedDataset(CommonTSDataset):
    def __init__(
        self,
        path: str,
        size: SequenceSize,
        data_split: DataSplit,
        split_sizes: Optional[DataSplitSize] = None,
        scale: bool = True,
        patch_size: int = 24,
    ):
        super(PatchAlignedDataset, self).__init__(
            path=path,
            size=size,
            data_split=data_split,
            split_sizes=split_sizes,
            scale=scale,
        )
        self.patch_size = patch_size

    def __get_item__(self, idx: int):
        data_split = self._get_data_splits()[self.data_split]
        src_start = self.patch_size - (idx + data_split.start) % self.patch_size
        src_start = idx + src_start
        source_slice = slice(src_start, src_start +
                             self.source_len - self.patch_size)

        overlap_len = self.overlap_len - self.overlap_len % self.patch_size
        pred_len = self.pred_len - self.pred_len % self.patch_size
        target_slice = slice(source_slice.stop - overlap_len,
                             source_slice.stop + pred_len)

        return (
            self.data[source_slice],
            self.data[target_slice],
            self.date_data[source_slice],
            self.date_data[target_slice],
            idx
        )


class DatasetLoaders(Enum):
    etth = ('etth', ETTHDataset)
    ettm = ('ettm', ETTMDataset)
    common = ('common', CommonTSDataset)
    context = ('context', ContextDataset)
    change_point = ('change_point', ChangePointDataset)
    patch_aligned = ('patch_aligned', PatchAlignedDataset)

    def __init__(self, key: str, config: Type[CommonTSDataset]):
        self.key = key
        self.config = config


DATASET_LOADER_KEY_MAP: Dict[str, Type[CommonTSDataset]] = {
    loader.key: loader.config for loader in DatasetLoaders
}
