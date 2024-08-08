from typing import Tuple
import scipy.stats as stats
import pandas as pd
import numpy as np
from plotly import express as px
from plotly import graph_objects as go
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, kpss


def filter_consecutive_duplicates(data):
    return [data[i] for i in range(1, len(data)) if data[i] != (data[i-1] + 1)]


def generate_series_cps(data: np.ndarray, patch_len: int = 36) -> list:
    last_stat = 0
    i = patch_len

    data_len = len(data)
    cps = []

    p_bar = tqdm(total=data_len)

    while i < data_len:
        cur_seq = data[last_stat: i].copy()
        cur_seq[0] += 1e-6
        _, adf_p_val, *_ = adfuller(cur_seq.copy(), autolag='AIC')
        _, kpss_p_val, *_ = kpss(cur_seq.copy(), regression='c', nlags='auto')
        if adf_p_val > 0.05 and kpss_p_val <= 0.05:
            cps.append(i)
            last_stat = i
            i += patch_len
            p_bar.update(patch_len)
        else:
            i += 1
            p_bar.update(1)
        p_bar.set_description_str(f"N of cps: {len(cps)}")

    return cps


def generate_cps(data: np.ndarray, patch_len: int = 36) -> np.ndarray:
    cps = np.zeros_like(data)

    for channel in range(data.shape[-1]):
        channel_cps = generate_series_cps(data[:, channel], patch_len)
        channel_cps = filter_consecutive_duplicates(channel_cps)
        cps[channel_cps, channel] = 1

    return cps


def normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit_transform(df)

    return df, scaler


def denormalize(df: pd.DataFrame, scaler: StandardScaler):
    return scaler.inverse_transform(df)


def cp_col_name(col: str) -> str:
    return f"{col}_cp"


def merge_df_with_cps(df: pd.DataFrame, cps: np.ndarray) -> pd.DataFrame:
    df_cols = list(df.columns)
    df_cols = [cp_col_name(col) for col in df_cols]

    df_cp = pd.DataFrame(cps, columns=df_cols)
    df = pd.concat([df_cp, df], axis=1)

    return df


def vline(pos, color=None, max_y: float = 1, min_y: float = 0):
    return {
        'type': 'line',
        'xref': 'x',
        'yref': 'y',
        'x0': pos,
        'y0': min_y,
        'x1': pos,
        'y1': max_y,
        'line': {
            "color": color,
            "width": 2,
        }}


def visualize_data(df: pd.DataFrame, data_col: str):
    data = df[data_col]

    max_y = data.max()
    min_y = data.min()

    graph = px.line(data, title='Original Data')

    lines = df.index[df[cp_col_name(data_col)] == 1].tolist()
    lines = [vline(i, 'red', max_y, min_y) for i in lines]

    graph.update_layout(shapes=lines)
    return graph


def app(filepath: str):
    df = pd.read_csv(filepath)

    df_cols = list(df.columns)
    has_date = 'date' in df_cols
    date_df = None
    if has_date:
        date_df = df['date']
        df.drop(columns=['date'], inplace=True)

    data, scaler = normalize(df)

    patch_len = 36
    cps = generate_cps(data.values, patch_len)

    scaler.inverse_transform(data)

    df = merge_df_with_cps(df, cps)

    if has_date:
        df['date'] = date_df

    updated_filepath = filepath.replace('.csv', '_cp.csv')
    df.to_csv(updated_filepath, index=False)

    graph = visualize_data(df, 'OT')
    graph.show()


if __name__ == '__main__':
    app('../data/ETTm1.csv')
