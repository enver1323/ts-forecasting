from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

dataset = 'exchange_rate'

df = pd.read_csv(f'data/{dataset}.csv')
data = df['OT'].values
start = 800
end = start+336
data = data[start:end]

def moving_average(x, w):
    start_padding = np.ones(w // 2) * x[0]
    end_padding = np.ones(w // 2) * x[-1]
    x = np.concatenate([start_padding, x, end_padding])
    return np.convolve(x, np.ones(w), 'valid') / w

def revin(x):
    x[150:] = ((x[150:] - x[150:].mean()) / x[150:].std()) * x[:150].std() + x[:150].mean()
    return (x - x.mean()) / x.std()

trend = moving_average(data, 25)
seasonality = data - trend
norm_x = revin(data)
matplotlib.rcParams.update({'font.size': 15})
# plt.plot(trend, label=dataset)
# plt.plot(seasonality, label=dataset)
# plt.plot(data, label=dataset)
plt.plot(norm_x, label=dataset)
plt.tight_layout()
plt.savefig(f'{dataset}_norm.png')
