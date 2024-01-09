from typing import Callable, Type, Optional, Sequence
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

from domain.point_id_ar import PointIDARTrainer
from domain.point_predictor.modules.model import PointPredictor, compute_loss
from domain.point_predictor.config import PointPredictorConfig
from generics import BaseConfig


class PointPredictorTrainer(PointIDARTrainer):
    def __init__(self, config: BaseConfig, device: torch.device):
        self.config: PointPredictorConfig
        self.model: PointPredictor
        super(PointPredictorTrainer, self).__init__(config, device)

    @property
    def model_type(self) -> Type[PointPredictor]:
        return PointPredictor

    def get_experiment_key(self, config: PointPredictorConfig):
        model_name = str(self.model)
        model_name = model_name[:model_name.find('(')]
        context_size = config.model.context_size
        initial_size = config.model.initial_size
        window_size = config.model.window_size
        data = config.data.dataset.path.split('/')[-1].split('.')[0]

        return f"{model_name}_{data}_{context_size}_({initial_size}->{window_size})_RNN_lag_normalized"

    @property
    def criterion(self) -> Callable:
        return compute_loss

    def _get_optimizers(self) -> Sequence[torch.optim.Optimizer]:
        return (
            torch.optim.AdamW(
                self.model.parameters(), self.config.lr.init
            ),
        )

    def visualize(self, batch: Sequence[torch.Tensor], filepath: str):
        file_dir = '/'.join(filepath.split('/')[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        with torch.no_grad():
            x, *_ = batch
            x = x.float().to(self.device)

            b, c = 0, -1

            trend_pts, season_pts, _, _ = self.model(x)  # C, B, Pts
            # trend_pts, trend = self.model(x)  # C, B, Pts
            trend, season = self.model.decomposition(x)
            x = x.detach().cpu()[b, :, c]

            def process_pred(x): return [
                item.detach().cpu() for item in x[c][b]
            ]

            offset = self.config.model.initial_size

            trend_pts = [pt.item() - offset for pt in process_pred(trend_pts)]
            season_pts = [
                pt.item() - offset for pt in process_pred(season_pts)]

            # trend = torch.concat([pred_slice for pred_slice in process_pred(trend)]).tolist()
            # season = torch.concat([pred_slice for pred_slice in process_pred(season)]).tolist()
            trend = trend.detach().cpu()[b, offset:, c]
            season = season.detach().cpu()[b, offset:, c]

            min_y = min(x.min().item(), min(trend), min(season))
            max_y = max(x.max().item(), max(trend), max(season))
            # min_y = min(x.min().item(), min(trend))
            # max_y = max(x.max().item(), max(trend))

            plt.plot(trend, label="Trend", color="orange")
            plt.vlines(x=trend_pts, ymin=min_y, ymax=max_y,
                       color='red', label='Trend pts')

            plt.plot(season, label="Season", color="green")
            plt.vlines(x=season_pts, ymin=min_y, ymax=max_y,
                       color='blue', label='Season pts', linewidth=2)

            plt.tight_layout()
            plt.legend(loc="lower left")
            plt.savefig(filepath)
            plt.close()
