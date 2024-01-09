import scipy.stats as stats
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

orig_df = pd.read_csv('../data/ETTh1.csv')
orig_df.drop(columns=['date'], inplace=True)
scaler = StandardScaler()
orig_df = scaler.fit_transform(orig_df)

df = orig_df[:, -1]
window_size = 64
segment_size = window_size // 2
df_len = len(df) - window_size + 1
cps = []
var_cps = []
mean_cps = []

min_y, max_y = min(df), max(df)


def vline(pos, color=None):
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


graph = px.line(df, title='Original Data')
print('Formed a graph')
for i in tqdm(range(df_len)):
    window = df[i:i+window_size]
    segment1 = window[:segment_size]
    segment2 = window[segment_size:]
    kl_div = stats.entropy(segment1, segment2)

    mean_diff = abs(segment1.mean() - segment2.mean())
    std_diff = abs(segment1.std() - segment2.std())

    if kl_div > 0.1:
        cps.append(vline(i+segment_size, None))

    if mean_diff > 0.04:
        mean_cps.append(vline(i+segment_size, 'MediumPurple'))

    if std_diff > 0.08:
        var_cps.append(vline(i+segment_size, 'olive'))


def filter_duplicate_lines(data):
    return [data[i]for i in range(1, len(data)) if data[i]['x0'] != (data[i-1]['x0'] + 1)]


cps = filter_duplicate_lines(cps)
graph.update_layout(shapes=cps)

var_cps = filter_duplicate_lines(var_cps)
graph.update_layout(shapes=var_cps)

mean_cps = filter_duplicate_lines(mean_cps)
graph.update_layout(shapes=mean_cps)

print('COMPLETED')
graph.show()
