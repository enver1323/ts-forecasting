from abc import ABC
from equinox import nn
import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jrandom
import jax.numpy as jnp
from domain.slider.modules.layers import Predictor, LinearSlider
from domain.slider.config import SliderConfig


class SlidingPredictor(eqx.Module):
    detail_slider: LinearSlider
    general_slider: LinearSlider
    predictor: Predictor

    def __init__(
        self,
        config: SliderConfig.ModelConfig,
        *,
        key: jrandom.PRNGKey
    ):
        super(SlidingPredictor, self).__init__()

        (detail_k, general_k, predictor_k) = jrandom.split(key, 3)

        self.detail_slider = LinearSlider(
            c_in=config.n_channels,
            kernel_size=config.patch_size,
            decomposition_kernel_size=config.kernel_size,
            stride=1,
            out_dim=config.d_model,
            key=detail_k
        )
        self.general_slider = LinearSlider(
            c_in=config.n_channels,
            kernel_size=config.patch_size,
            decomposition_kernel_size=config.kernel_size,
            stride=config.stride,
            out_dim=config.d_model,
            key=general_k
        )
        self.predictor = Predictor(
            in_feat_size=2*config.d_model, hid_feat_size=config.d_inner, out_feat_size=config.n_channels,
            in_seq_size=config.patch_size, hid_seq_size=config.patch_size, out_seq_size=config.pred_len,
            key=predictor_k
        )

    def __call__(self, x: jnp.ndarray):
        x_detail = self.detail_slider(x)
        x_general = self.general_slider(x)

        x = x_detail * x_general

        x = self.predictor(x)
        return x