import jax
import jax.numpy as jnp
import jax.random as jrandom
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import math
from functools import partial


@partial(jax.jit, static_argnames=['kernel_size'])
def moving_avg(data: jnp.ndarray, kernel_size: int) -> jnp.ndarray:
    target_len, dim_size = data.shape
    pad_shape = ((kernel_size // 2, math.ceil(kernel_size / 2) - 1), (0, 0))
    padded_data = jnp.pad(data, pad_shape, 'edge')

    def _avg(_, i):
        slice_x = (i, 0)
        slice_y = (kernel_size, dim_size)
        sliced_data = jax.lax.dynamic_slice(padded_data, slice_x, slice_y)
        sliced_data = sliced_data.mean(axis=0)
        return None, sliced_data

    _, averaged_data = jax.lax.scan(_avg, None, jnp.arange(0, target_len))

    averaged_data = jnp.stack(averaged_data)

    return averaged_data

# @partial(jax.jit)
# def critical_points(data: jnp.ndarray) -> jnp.ndarray:
#     target_len, dim_size = data.shape

#     def _avg(_, i):
#         slice_x = (i, 0)
#         slice_y = (kernel_size, dim_size)
#         sliced_data = jax.lax.dynamic_slice(padded_data, slice_x, slice_y)
#         sliced_data = sliced_data.mean(axis=0)
#         return None, sliced_data

#     _, averaged_data = jax.lax.scan(_avg, None, jnp.arange(0, target_len))

#     averaged_data = jnp.stack(averaged_data)

#     return averaged_data


def app():
    dataset = pd.read_csv('./data/ETTh1.csv')
    dataset = dataset.drop(columns=['date'])

    scaler = StandardScaler()

    all_data = scaler.fit_transform(dataset.values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=all_data[:, -1]))

    kernel_size = 100
    ma = moving_avg(all_data, kernel_size)
    fig.add_trace(go.Scatter(y=ma[:, -1]))
    fig.add_trace(go.Scatter(y=(all_data[:, -1] - ma[:, -1])))

    mama = moving_avg(ma, 2 * kernel_size)
    fig.add_trace(go.Scatter(y=mama[:, -1]))
    fig.add_trace(go.Scatter(y=(ma[:, -1] - mama[:, -1])))


    fig.show()


if __name__ == '__main__':
    app()