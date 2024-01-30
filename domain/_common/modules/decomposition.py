import math
from typing import Tuple

import torch
from torch import nn, Tensor


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