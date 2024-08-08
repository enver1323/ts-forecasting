import torch
from torch import Tensor
from torch.nn import functional as F
import math


def kl_divergence(prediction: Tensor, ground_truth: Tensor) -> Tensor:
    return -(ground_truth * prediction.log() + (1 - ground_truth) * (1 - prediction.log())).mean()


def uni_kl_divergence(x_mean: Tensor, x_std: Tensor, y_mean: Tensor, y_std: Tensor) -> Tensor:
    return (x_std / y_std).log() + (x_std ** 2 + (x_mean - y_mean) ** 2) / 2 * y_std ** 2 - 1 / 2


def mse(prediciton: Tensor, ground_truth: Tensor) -> Tensor:
    return ((prediciton - ground_truth) ** 2).mean()


def mae(prediction: Tensor, ground_truth: Tensor, axes=None) -> Tensor:
    return (abs(prediction - ground_truth)).mean(axes)


def lag_loss(prediction: Tensor, dim: int = -1) -> Tensor:
    start_slices = [slice(None)] * prediction.ndim
    start_slices[dim] = slice(1, None)
    end_slices = [slice(None)] * prediction.ndim
    end_slices[dim] = slice(None, -1)

    diff = prediction[start_slices] - prediction[end_slices]
    diff = diff[start_slices] - diff[end_slices]

    return diff.abs().mean()


def log_cosh_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    def _log_cosh(x: Tensor) -> Tensor:
        return x + F.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))
