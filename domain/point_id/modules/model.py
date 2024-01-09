import torch
from torch import nn, Tensor
from typing import Sequence, Type
from domain._common.utils import get_channelwise_modules
from domain._common.modules.normalization import RevIN
from domain._common.modules.decomposition import SeriesDecomposition, MovingAverage
from domain.point_id.modules.layers import TrendPointEstimator, SeasonPointEstimator
from domain.point_id.config import PointIDConfig


class PointID(nn.Module):
    def __init__(
            self,
            config: PointIDConfig.ModelConfig,
    ):
        super(PointID, self).__init__()

        self.n_channels = config.n_channels
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_pts = config.n_points

        point_estimator_args = (self.seq_len, self.pred_len, self.n_pts)
        self.trend_pts = get_channelwise_modules(
            self.n_channels, TrendPointEstimator, point_estimator_args)
        self.season_pts = get_channelwise_modules(
            self.n_channels, SeasonPointEstimator, point_estimator_args)
        # self.season_lins = get_channelwise_modules(
        #     self.n_channels, nn.Linear, (self.seq_len, self.pred_len))

        self.decomposition = SeriesDecomposition(
            MovingAverage(config.kernel_size))

        self.rev_in = RevIN(1)

    def _get_channelwise_modules(self, module: Type[nn.Module], args: Sequence) -> nn.ModuleList:
        return nn.ModuleList([
            module(*args)
            for i in range(self.n_channels)
        ])

    def forward(self, *args: Tensor) -> Tensor:
        x, *_ = args    # (B, L, D)
        x = self.rev_in(x, 'norm')
        trend: Tensor
        season: Tensor
        trend, season = self.decomposition(x)  # (B, L, D), (B, L, D)

        trend = trend.transpose(-1, -2)  # (D, L)
        season = season.transpose(-1, -2)  # (D, L)

        result = []
        for c in range(self.n_channels):
            cur_trend = self.trend_pts[c](trend[:, c, :])
            cur_season = self.season_pts[c](season[:, c, :])
            # cur_season = self.season_lins[c](season[:, c, :])
            cur_x = cur_trend + cur_season
            result.append(cur_x)

        result = torch.stack(result)
        result = result.permute(1, 2, 0)

        result = self.rev_in(result, 'denorm')

        return result
