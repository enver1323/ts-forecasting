from domain._common.modules.decomposition_jax import EMA, MovingAverage
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

dataset = 'ETTh1'

df = pd.read_csv(f'data/{dataset}.csv')
data = df['OT'].values
start = 824
end = start+336
data = data[start:end]

def ema(x):
    module = EMA(0.05)
    return module(x)

def sma(x):
    module = MovingAverage(25)
    return module(x)

trend_sma = sma(np.expand_dims(data, 0))[0]
trend_ema = ema(data)
seasonality_sma = data - trend_sma
seasonality_ema = data - trend_ema
# norm_x = revin(data)
matplotlib.rcParams.update({'font.size': 15})
plt.plot(trend_sma, label=f"Trend SMA {dataset}")
plt.plot(trend_ema, label=f"Trend EMA {dataset}")
plt.plot(seasonality_sma, label=f"Seasonality SMA {dataset}")
plt.plot(seasonality_ema, label=f"Seasonality EMA {dataset}")
# plt.plot(data, label=f"Data {dataset}")
# plt.plot(norm_x, label=dataset)
plt.legend()
plt.tight_layout()
plt.savefig(f'{dataset}.png')
