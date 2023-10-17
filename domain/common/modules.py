import torch
from torch import nn, Tensor
from typing import Tuple, Type, Sequence
import math


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=stride, padding=0
        )

    def forward(self, data: Tensor) -> Tensor:
        front = data[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = data[:, -1:, :].repeat(1, math.ceil(self.kernel_size / 2) - 1, 1)

        x = torch.cat([front, data, end], dim=-2)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, decomposer: nn.Module):
        super(SeriesDecomposition, self).__init__()

        self.decomposer = decomposer

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        trend = self.decomposer(data)
        seasonality = data - trend
        return trend, seasonality


def get_channelwise_modules(n_channels: int, module: Type[nn.Module], args: Sequence) -> nn.ModuleList:
    return nn.ModuleList([
        module(*args)
        for i in range(n_channels)
    ])


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
