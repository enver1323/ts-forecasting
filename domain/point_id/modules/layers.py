import torch
from torch import nn, Tensor
from abc import abstractmethod, ABC
from typing import Callable
import numpy as np
import math


class PointEstimator(nn.Module, ABC):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_pts: int,
        n_params: int
    ):
        super(PointEstimator, self).__init__()
        self.pred_len = pred_len
        self.n_params = n_params
        self.n_pts = n_pts
        self._n_slots = self.n_pts + 1

        self.pts_lin = nn.Linear(seq_len, self.n_pts * pred_len)
        self.component_lin = nn.Linear(seq_len, self._n_slots * self.n_params)
        self.combine_lin = nn.Linear(self.pred_len, 1)

        self.__init_weights_uniform(self.component_lin)

    def __init_weights_uniform(self, module: nn.Linear):
        n = module.in_features
        y = 1.0/np.sqrt(n)
        module.weight.data.uniform_(-y, y)
        module.bias.data.fill_(0)

    def _generate_points(self, x: Tensor) -> Tensor:
        x = self.pts_lin(x)
        x = x.reshape(x.shape[0], self.n_pts, self.pred_len)
        x = x.argmax(dim=-1)
        x, _ = x.sort(dim=-1)

        return x

    def _generate_component_params(self, x: Tensor) -> Tensor:
        x = self.component_lin(x)
        x = x.reshape(x.shape[0], self._n_slots, self.n_params)
        return x

    def _get_pts_mask(self, pts_data: Tensor, condition: Callable[[Tensor], Tensor]) -> Tensor:
        mask = torch.arange(self.pred_len).reshape(
            1, 1, -1).to(pts_data.device)
        mask = mask.repeat(pts_data.shape[0], pts_data.shape[1], 1)
        mask = torch.where(condition(mask), 1, 0)
        return mask

    def _get_pts_distances(self, pts: Tensor) -> Tensor:
        B = pts.shape[0]
        pts_template = torch.ones((B, 1)).to(pts.device)
        pts_start_shifted = torch.concat([pts_template * 0, pts], dim=-1)
        pts_end_shifted = torch.concat(
            [pts, self.pred_len * pts_template], dim=-1)

        return pts_end_shifted - pts_start_shifted
    
    def _get_pts_scale(self, pts: Tensor) -> Tensor:
        pts_scale = self._get_pts_distances(pts).unsqueeze(-1) + 1
        pts_scale = pts_scale / self._n_slots
        pts_scale = pts_scale + 1e-4

        return pts_scale


    def _combine_by_pts(self, pts_data: Tensor, pts: Tensor) -> Tensor:
        B = pts.shape[0]

        pts = torch.concat((
            torch.zeros((B, 1)).to(pts.device),
            pts,
            (torch.ones((B, 1)) * self.pred_len).to(pts.device)
        ), dim=-1)  # (B, n_pts,)

        indices = torch.arange(self._n_slots)

        # TODO: Test sigmoid
        # TODO: Test [0:end] instead of start:end
        start_ids = pts[:, indices].int().reshape(B, -1, 1)
        end_ids = pts[:, indices + 1].int().reshape(B, -1, 1)

        pts_mask = self._get_pts_mask(
            pts_data,
            lambda mask: (mask >= start_ids) & (mask < end_ids)
        )
        result = (pts_data) * pts_mask
        result = result.sum(-2)

        return result

    def _combine_by_pts_with_sum(self, pts_data: Tensor, pts: Tensor) -> Tensor:
        weights: Tensor = self.combine_lin(pts_data)
        weights = torch.softmax(weights, -2).transpose(-1, -2)

        result = weights @ pts_data
        result = result.squeeze(-2)

        return result

    @abstractmethod
    def _generate(self, params: Tensor, pts: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor) -> Tensor:
        pts = self._generate_points(x)  # (B, n_pts,)
        # (B, _n_slots, _n_trend_params)
        params = self._generate_component_params(x)
        x = self._generate(params, pts)  # (B, pred_len,)

        return x


class TrendPointEstimator(PointEstimator):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_pts: int
    ):
        super(TrendPointEstimator, self).__init__(seq_len, pred_len, n_pts, 2)

    def _combine_by_pts(self, pts_data: Tensor, pts: Tensor) -> Tensor:
        B = pts.shape[0]

        pts = torch.concat((
            torch.zeros((B, 1)).to(pts.device),
            pts,
            (torch.ones((B, 1)) * self.pred_len).to(pts.device)
        ), dim=-1)  # (B, n_pts,)

        indices = torch.arange(self._n_slots)

        # TODO: Test sigmoid
        # TODO: Test [0:end] instead of start:end
        start_ids = pts[:, indices].int().reshape(B, -1, 1)
        end_ids = pts[:, indices + 1].int().reshape(B, -1, 1)

        pts_mask = self._get_pts_mask(
            pts_data,
            lambda mask: (mask >= start_ids) & (mask < end_ids)
        )  # (B, _n_slots, pred_len)
        start_pts_mask = self._get_pts_mask(
            pts_data,
            lambda mask: (mask == start_ids)
        )  # (B, _n_slots, pred_len)
        end_pts_mask = self._get_pts_mask(
            pts_data,
            lambda mask: (mask == (end_ids - 1))
        )  # (B, _n_slots, pred_len)

        start_pts_data = (
            pts_data * start_pts_mask).sum(dim=-1).unsqueeze(dim=-1)
        start_pts_data[:, 0, :] = 0
        end_pts_data = (pts_data * end_pts_mask).sum(dim=-1).unsqueeze(dim=-1)
        end_pts_data = torch.concat((
            torch.zeros((B, 1, 1)).to(end_pts_data.device),
            end_pts_data[:, :-1, :]
        ), dim=-2).to(end_pts_data.device)

        result = (pts_data - start_pts_data + end_pts_data) * pts_mask
        result = result.sum(-2)

        return result

    def _generate(self, params: Tensor, pts: Tensor) -> Tensor:
        x = torch.arange(self.pred_len).reshape(
            (1, 1, self.pred_len)).float().to(params.device)
        
        a, b = params[:, :, 0:1], params[:, :, 1:2]  # (B, _n_slots, 1)
        a = a / self._get_pts_scale(pts)
        b = b / self._get_pts_scale(pts) - 1

        y = a @ x + b  # (B, _n_slots, pred_len)

        combined = self._combine_by_pts(y, pts)
        return combined


class SeasonPointEstimator(PointEstimator):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_pts: int,
    ):
        super(SeasonPointEstimator, self).__init__(
            seq_len, pred_len, n_pts, pred_len)
        # super(SeasonPointEstimator, self).__init__(
        #     seq_len, pred_len, n_pts, 3)

    def _generate_lin(self, params: Tensor, pts: Tensor) -> Tensor:
        y = params.reshape(params.shape[0], self._n_slots, self.pred_len)

        combined = self._combine_by_pts(y, pts)
        return combined

    def _generate_sin(self, params: Tensor, pts: Tensor) -> Tensor:
        x = torch.arange(self.pred_len)\
            .reshape((1, 1, self.pred_len)).float().to(params.device)
        a, b, c = params[:, :, 0:1], params[:, :,
                                            1:2], params[:, :, 2:3]  # (B, _n_slots, 1)
        y = a * torch.sin(b * x + c)  # (B, _n_slots, pred_len)

        combined = self._combine_by_pts(y, pts)
        return combined

    def _generate(self, params: Tensor, pts: Tensor) -> Tensor:
        y = params / self._get_pts_scale(pts)

        combined = self._combine_by_pts(y, pts)
        return combined
