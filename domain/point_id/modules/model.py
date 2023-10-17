import torch
from torch import nn, Tensor
from typing import Tuple, Dict, Any, Sequence, Type
from domain.common.dilate_loss.soft_dtw import SoftDTW
from domain.common.dilate_loss.dilate import dilate_loss
from domain.common.modules import SeriesDecomposition, MovingAverage, get_channelwise_modules
from domain.common.metrics import mse, mae, kl_divergence
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
        self.season_lins = get_channelwise_modules(
            self.n_channels, nn.Linear, (self.seq_len, self.pred_len))

        self.decomposition = SeriesDecomposition(
            MovingAverage(config.kernel_size))

    def _get_channelwise_modules(self, module: Type[nn.Module], args: Sequence) -> nn.ModuleList:
        return nn.ModuleList([
            module(*args)
            for i in range(self.n_channels)
        ])

    def forward(self, *args: Tensor) -> Tensor:
        x, *_ = args    # (B, L, D)
        trend: Tensor
        season: Tensor
        trend, season = self.decomposition(x)  # (B, L, D), (B, L, D)

        trend = trend.transpose(-1, -2)  # (D, L)
        season = season.transpose(-1, -2)  # (D, L)

        result = []
        for c in range(self.n_channels):
            cur_trend = self.trend_pts[c](trend[:, c, :])
            # cur_season = self.season_pts[c](season[:, c, :])
            cur_season = self.season_lins[c](season[:, c, :])
            cur_x = cur_trend + cur_season
            result.append(cur_x)

        result = torch.stack(result)
        result = result.permute(1, 2, 0)

        return result


def compute_loss(model: PointID, *args: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
    x, y, *_ = args

    y = y[:, -model.pred_len:, :]
    pred_y = model(x)

    # loss, soft_dtw_loss, path_dtw_loss = dilate_loss(pred_y, y, normalize=True)
    # loss = loss.mean()
    # soft_dtw_loss = soft_dtw_loss.mean()
    # path_dtw_loss = path_dtw_loss.mean()

    soft_dtw_loss = SoftDTW(normalize=False)(pred_y, y)
    loss = soft_dtw_loss = soft_dtw_loss.mean()

    mse_loss = mse(pred_y, y)

    loss = loss + 800 * mse_loss

    metrics = {
        "Loss": loss.item(),
        "SoftDTW": soft_dtw_loss.item(),
        # "PathDTW": path_dtw_loss.item(),
        "MSE": mse_loss.item(),
        "MAE": mae(pred_y, y).item(),
    }

    return loss, metrics
