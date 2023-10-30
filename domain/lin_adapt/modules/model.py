import torch
from torch import nn, Tensor
from typing import Tuple, Dict, Any, Sequence, Type
from domain.common.dilate_loss.soft_dtw import SoftDTW
from domain.common.dilate_loss.dilate import dilate_loss
from domain.common.modules import SeriesDecomposition, MovingAverage, get_channelwise_modules, AdaptiveNorm
from domain.common.metrics import mse, mae, kl_divergence
from domain.lin_adapt.modules.layers import ComponentLinAdapt
from domain.lin_adapt.config import LinAdaptConfig


class LinAdapt(nn.Module):
    def __init__(
            self,
            config: LinAdaptConfig.ModelConfig,
    ):
        super(LinAdapt, self).__init__()

        self.n_channels = config.n_channels
        seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.individual = config.individual

        if self.individual:
            lin_args = (seq_len, self.pred_len)
            self.trend_lins = get_channelwise_modules(
                self.n_channels, ComponentLinAdapt, lin_args)
            self.season_lins = get_channelwise_modules(
                self.n_channels, ComponentLinAdapt, lin_args)
        else:
            self.trend_lin = ComponentLinAdapt(seq_len, self.pred_len)
            self.season_lin = ComponentLinAdapt(seq_len, self.pred_len)

        self.decomposition = SeriesDecomposition(
            MovingAverage(config.kernel_size))

    def forward(self, *args: Tensor) -> Tensor:
        x, *_ = args    # (B, L, D)

        trend: Tensor
        season: Tensor
        trend, season = self.decomposition(x)  # (B, L, D), (B, L, D)

        trend = trend.transpose(-1, -2)  # (B, D, L)
        season = season.transpose(-1, -2)  # (B, D, L)

        if self.individual:
            result = []
            for c in range(self.n_channels):
                cur_trend = self.trend_lins[c](trend[:, c, :])
                cur_season = self.season_lins[c](season[:, c, :])

                cur_x = cur_trend + cur_season
                result.append(cur_x)

            result = torch.stack(result)
            result = result.permute(1, 2, 0)
        else:
            cur_trend = self.trend_lin(trend)
            cur_season = self.season_lin(season)
            result = cur_trend + cur_season
            result = result.transpose(-1, -2)

        return result


def compute_norm_loss(model: LinAdapt, x: Tensor, y: Tensor) -> Tensor:
    trend_y: Tensor
    season_y: Tensor
    with torch.no_grad():
        trend_y, season_y = model.decomposition(y)

    model(x)

    if model.individual:
        pred_trend_means = torch.stack(
            [lin.norm.mean for lin in model.trend_lins])
        pred_trend_stds = torch.stack(
            [lin.norm.std for lin in model.trend_lins])
        pred_season_means = torch.stack(
            [lin.norm.mean for lin in model.season_lins])
        pred_season_stds = torch.stack(
            [lin.norm.std for lin in model.season_lins])
    else:
        pred_trend_means = model.trend_lin.norm.mean
        pred_season_means = model.season_lin.norm.mean
        pred_trend_stds = model.trend_lin.norm.std
        pred_season_stds = model.season_lin.norm.std

    trend_mean_loss = kl_divergence(
        pred_trend_means, trend_y.mean(dim=-2, keepdim=True).detach())
    season_mean_loss = kl_divergence(
        pred_season_means, season_y.mean(dim=-2, keepdim=True).detach())

    trend_std_loss = kl_divergence(
        pred_trend_stds, trend_y.var(dim=-2, keepdim=True).sqrt().detach())
    season_std_loss = kl_divergence(
        pred_season_stds, season_y.var(dim=-2, keepdim=True).sqrt().detach())

    loss = trend_mean_loss + season_mean_loss + trend_std_loss + season_std_loss
    return loss


def compute_loss(model: LinAdapt, *args: Tensor) -> Tuple[Tuple[Tensor, Tensor, Tensor], Dict[str, Any]]:
    x, y, *_ = args

    y = y[:, -model.pred_len:, :]
    pred_y = model(x)

    # loss, soft_dtw_loss, path_dtw_loss = dilate_loss(pred_y, y, normalize=True)
    # loss = loss.mean()
    # soft_dtw_loss = soft_dtw_loss.mean()
    # path_dtw_loss = path_dtw_loss.mean()

    soft_dtw_loss = SoftDTW(normalize=False)(pred_y, y)
    soft_dtw_loss = soft_dtw_loss.mean()

    pred_y = model(x)

    mse_loss = mse(pred_y, y)
    mse_scale = 100  # 200

    # norm_loss = compute_norm_loss(model, x, y)

    loss = soft_dtw_loss + mse_scale * mse_loss

    metrics = {
        "Loss": loss.item(),
        "SoftDTW": soft_dtw_loss.item(),
        # "PathDTW": path_dtw_loss.item(),
        "MSE": mse_loss.item(),
        "MAE": mae(pred_y, y).item(),
    }

    return (mse_loss, soft_dtw_loss, mse_loss), metrics
