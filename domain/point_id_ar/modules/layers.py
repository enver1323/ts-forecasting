import torch
from torch import nn, Tensor
import math
from abc import abstractmethod, ABC
from domain._common.modules.normalization import RevIN
from typing import Callable, List, Union, Sequence


class PointEstimator(nn.Module, ABC):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        window_len: int,
        n_choices: int,
        n_params: int
    ):
        super(PointEstimator, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_params = n_params
        self.n_choices = n_choices

        # pts_max_len = window_len - 1
        pts_max_len = self.pred_len - 1
        pts_hidden = (self.seq_len + pts_max_len) // 2
        self.pts_id = nn.Sequential(
            nn.Linear(self.seq_len, pts_hidden),
            nn.GELU(),
            nn.Linear(pts_hidden, pts_max_len)
        )
        # self.pts_id = nn.Linear(self.seq_len, self.pred_len - 1)

        # component_hidden = (self.seq_len + self.n_choices * self.n_params) // 2
        # self.component_lin = nn.Sequential(
        #     nn.Linear(self.seq_len, component_hidden),
        #     nn.GELU(),
        #     nn.Linear(component_hidden, self.n_choices * self.n_params)
        # )
        self.component_lin = nn.Linear(
            self.seq_len, self.n_choices * self.n_params)

        self.select_lin = nn.Linear(self.pred_len, 1)

        self.res_lin = nn.Linear(self.seq_len, self.pred_len)
        self.res_scale = nn.Parameter(torch.zeros((self.pred_len)))
        # self.rev_in = RevIN(1)

    def _generate_points(self, x: Tensor) -> Tensor:
        x = self.pts_id(x)
        x = x.argmax(dim=-1) + 1

        return x

    def _generate_component_params(self, x: Tensor) -> Tensor:
        x = self.component_lin(x)
        x = x.reshape(x.shape[0], self.n_choices, self.n_params)

        return x

    def _get_pts_mask(self, pts_data: Tensor, condition: Callable[[Tensor], Tensor]) -> Tensor:
        mask = torch.arange(self.pred_len).reshape(
            1, 1, -1).to(pts_data.device)
        mask = mask.repeat(pts_data.shape[0], pts_data.shape[1], 1)
        mask = torch.where(condition(mask), 1, 0)

        return mask

    def _combine_by_pts(self, pts_data: Tensor, pts: Tensor) -> Sequence[Tensor]:
        B = pts_data.shape[0]
        choice_weights: Tensor = self.select_lin(pts_data)  # B, n_choices
        choice_indices = torch.argmax(choice_weights.squeeze(-1), -1)  # B

        # pts_data = self.rev_in(pts_data, 'denorm')

        outputs = []

        for b_idx in range(B):
            choice = pts_data[b_idx, choice_indices[b_idx]]  # pred_len
            choice = choice[:pts[b_idx]]
            outputs.append(choice)

        return outputs

    @abstractmethod
    def _generate(self, params: Tensor, pts: Tensor) -> Union[Tensor, List[Tensor]]:
        pass

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        outputs = [x[idx] for idx in range(B)]
        total_len = self.seq_len + self.pred_len
        residual = self.res_lin(x)

        while any([len(output) < total_len for output in outputs]):
            x = torch.stack([output[-self.seq_len:]
                            for output in outputs if len(output) < total_len])

            # x = self.rev_in(x, 'norm')

            pts = self._generate_points(x)  # (B,)
            # (B, n_choices, n_params)
            params = self._generate_component_params(x)

            pts_preds = self._generate(params, pts)  # List[Tensor]

            pred_idx = 0
            for idx in range(B):
                if len(outputs[idx]) >= total_len:
                    continue

                cur_pred = pts_preds[pred_idx]
                outputs[idx] = torch.cat((outputs[idx], cur_pred))
                pred_idx += 1

        outputs = [output[self.seq_len:total_len] for output in outputs]
        output_tensor = torch.stack(outputs)

        output_tensor = (1 - self.res_scale) * \
            output_tensor + self.res_scale * residual

        return output_tensor


class TrendPointEstimator(PointEstimator):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        window_len: int,
        n_choices: int
    ):
        super(TrendPointEstimator, self).__init__(
            seq_len, pred_len, window_len, n_choices, 2)

    def _generate(self, params: Tensor, pts: Tensor) -> Sequence[Tensor]:
        x = torch.arange(self.pred_len).float().to(params.device)\
            .reshape((1, 1, self.pred_len))
        a, b = params[:, :, 0:1], params[:, :, 1:2]  # (B, n_choices, 1)
        a = a / pts.reshape(-1, 1, 1).sqrt() / math.sqrt(self.n_choices)

        y = a @ x + b  # (B, n_choices, pred_len)

        combined = self._combine_by_pts(y, pts)
        return combined


class SeasonPointEstimator(PointEstimator):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        window_len: int,
        n_choices: int,
    ):
        super(SeasonPointEstimator, self).__init__(
            seq_len, pred_len, window_len, n_choices, pred_len)
        # super(SeasonPointEstimator, self).__init__(
        #     seq_len, pred_len, n_choices, 3)

    def _generate(self, params: Tensor, pts: Tensor) -> Sequence[Tensor]:
        combined = self._combine_by_pts(params, pts)
        return combined
