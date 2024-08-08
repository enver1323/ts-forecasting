import torch
from torch import nn, Tensor
from domain._common.modules.normalization import RevIN
from domain._common.utils import get_channelwise_modules
from domain._common.modules.decomposition import SeriesDecomposition, MovingAverage
from domain.point_id_ar.modules.layers import TrendPointEstimator, SeasonPointEstimator
from domain.point_id_ar.config import PointIDARConfig


class PointIDAR(nn.Module):
    def __init__(
            self,
            config: PointIDARConfig.ModelConfig,
    ):
        super(PointIDAR, self).__init__()

        self.n_channels = config.n_channels
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        point_estimator_args = (self.seq_len, self.pred_len, config.window_len, config.n_choices)
        self.trend_pts = get_channelwise_modules(
            self.n_channels, TrendPointEstimator, point_estimator_args)
        self.season_pts = get_channelwise_modules(
            self.n_channels, SeasonPointEstimator, point_estimator_args)
        # self.season_lins = get_channelwise_modules(
        #     self.n_channels, nn.Linear, (self.seq_len, self.pred_len))

        self.decomposition = SeriesDecomposition(MovingAverage(config.kernel_size))
        self.rev_in = RevIN(self.n_channels)
        # self.transformer_enc = nn.TransformerEncoderLayer(d_model=self.pred_len, nhead=4, dim_feedforward=self.pred_len * 4, activation=nn.GELU(), batch_first=True)
        # self.attn = nn.MultiheadAttention(self.pred_len, 4, dropout=0.5, batch_first=True)


    def forward(self, *args: Tensor) -> Tensor:
        x, *_ = args    # (B, L, D)
        x = self.rev_in(x, 'norm')
        trend: Tensor
        season: Tensor
        trend, season = self.decomposition(x)  # (B, L, D), (B, L, D)

        trend = trend.transpose(-1, -2)  # (D, L)
        season = season.transpose(-1, -2)  # (D, L)
        # x = x.transpose(-1, -2)  # (D, L)

        result = []
        for c in range(self.n_channels):
            cur_trend = self.trend_pts[c](trend[:, c, :])
            cur_season = self.season_pts[c](season[:, c, :])
            # cur_season = self.season_lins[c](season[:, c, :])
            cur_x = cur_trend + cur_season
            # cur_x = self.season_pts[c](x[:, c, :])
            result.append(cur_x)

        result = torch.stack(result)

        result = result.permute(1, 2, 0)
        # result = result.permute(1, 0, 2)
        # result = self.transformer_enc(result)
        # result = result.permute(0, 2, 1)

        result = self.rev_in(result, 'denorm')

        return result