from torch import nn, Tensor
from typing import Tuple, Dict, Any, Sequence
from domain._common.losses.softdtw_loss.soft_dtw_torch import SoftDTW
from domain._common.utils import get_channelwise_modules
from domain._common.modules.decomposition import SeriesDecomposition, MovingAverage
from domain._common.losses.metrics_torch import lag_loss
from domain.point_predictor.modules.layers import ComponentPointPredictor
from domain.point_predictor.config import PointPredictorConfig


class PointPredictor(nn.Module):
    def __init__(
            self,
            config: PointPredictorConfig.ModelConfig,
    ):
        super(PointPredictor, self).__init__()

        self.n_channels = config.n_channels
        self.context_size = config.context_size
        initial_size = config.initial_size
        hidden_size = config.hidden_size
        window_size = config.window_size

        point_args = (self.context_size, initial_size,
                      hidden_size, window_size)
        self.trend_lins = get_channelwise_modules(
            self.n_channels, ComponentPointPredictor, point_args)
        self.season_lins = get_channelwise_modules(
            self.n_channels, ComponentPointPredictor, point_args)

        self.decomposition = SeriesDecomposition(
            MovingAverage(config.kernel_size))

    def forward(self, x: Tensor) -> Tuple[
        Sequence[Sequence[Tensor]], Sequence[Sequence[Tensor]],
        Sequence[Sequence[Tensor]], Sequence[Sequence[Tensor]]
    ]:
        trend = x
        trend: Tensor
        season: Tensor
        trend, season = self.decomposition(x)  # (B, L, D), (B, L, D)

        trend_pts = []
        trend_slices = []
        season_pts = []
        season_slices = []
        
        for c in range(self.n_channels):
            pts, slices = self.trend_lins[c](trend[:, :, c])
            trend_pts.append(pts) # C, B, Pts
            trend_slices.append(slices) # C, B, Pts
            
            pts, slices = self.season_lins[c](season[:, :, c])  # C, B, Pts
            season_pts.append(pts)
            season_slices.append(slices)

        return trend_pts, season_pts, trend_slices, season_slices
        # return trend_pts, trend_slices


def compute_component_loss(pred_slices: Sequence[Sequence[Tensor]], criterion: nn.Module) -> Tensor:
    total_loss = None
    
    C, B = len(pred_slices), len(pred_slices[0])

    for b in range(B):
        for c in range(C):
            slices = pred_slices[c][b]
            n_steps = len(slices) - 1
            for i in range(0, n_steps):
                criterion_shape = (1, -1, 1)
                
                loss: Tensor = criterion(
                    slices[i].reshape(criterion_shape),
                    slices[i+1].reshape(criterion_shape)
                )
                loss = -loss / n_steps
                
                total_loss = loss if total_loss is None else total_loss + loss

            loss = sum([lag_loss(pred) for pred in slices if len(pred) > 2]) / len(slices)
            total_loss = loss if total_loss is None else total_loss + loss

    total_loss = total_loss / C / B

    return total_loss


def compute_loss(model: PointPredictor, *args: Tensor) -> Tuple[Tuple[Tensor], Dict[str, Any]]:
    x, *_ = args
    _, _, trend_slices, season_slices = model(x)  # C, B, Pts
    # _, trend_slices = model(x)  # C, B, Pts

    criterion = SoftDTW(normalize=True)

    trend_loss = compute_component_loss(trend_slices, criterion)
    season_loss = compute_component_loss(season_slices, criterion)

    # loss = - (trend_loss + season_loss)
    loss = (trend_loss + season_loss)
    # loss = -trend_loss

    metrics = {
        "Loss": loss.item(),
        # "PathDTW": path_dtw_loss.item(),
    }
    
    losses = (
        loss,
    )

    return losses, metrics
