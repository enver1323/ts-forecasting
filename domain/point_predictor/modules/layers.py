from typing import List, Tuple, Sequence
import torch
from torch import nn, Tensor
from domain._common.utils import soft_argmax
from domain._common.modules.normalization import AdaptiveNorm


class ComponentPointPredictor(nn.Module):
    def __init__(
        self,
        context_size: int,
        initial_size: int,
        hidden_size: int,
        window_size: int,
    ):
        super(ComponentPointPredictor, self).__init__()

        self.context_size = context_size
        self.initial_size = initial_size
        self.hidden_size = hidden_size
        self.window_size = window_size - 1

        self.hid_lin = nn.Linear(self.hidden_size, self.hidden_size)
        self.in_lin = nn.Linear(self.initial_size, self.hidden_size)
        self.activation = nn.GELU()
        self.pred_lin = nn.Linear(self.hidden_size, self.window_size)

    def _get_pts_indices(self, pts: List[List[Tensor]]):
        B = len(pts)
        indices = [
            idx for idx in range(B)
            if len(pts[idx]) == 0 or pts[idx][-1].item() < self.context_size
        ]
        return indices

    def _get_slices_from_pts(self, x: Tensor, pts: List[List[Tensor]], pt_ids: Sequence[int]):
        slices = torch.stack([
            x[idx, pts[idx][-1].long()-self.initial_size:pts[idx][-1].long()]
            if len(pts[idx]) > 0 else
            x[idx, 0:self.initial_size]
            for idx in pt_ids
        ])
        return slices

    def forward(self, x: Tensor) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        B = x.shape[0]
        pts = [list() for _ in range(B)]
        x_slices = [list() for _ in range(B)]
        hidden = torch.ones((B, self.hidden_size)).to(x.device)

        def any_points_incomplete(): return len(pts[0]) == 0 or any(
            [pt[-1].item() < self.context_size for pt in pts]
        )

        while any_points_incomplete():
            pt_ids = self._get_pts_indices(pts)
            cur_slices = self._get_slices_from_pts(x, pts, pt_ids)
            cur_hidden = hidden[pt_ids, :]

            cur_hidden = torch.tanh(self.hid_lin(cur_hidden) + self.in_lin(cur_slices))
            hidden[pt_ids] = cur_hidden
            pts_weights = self.pred_lin(cur_hidden)

            cur_pts = soft_argmax(pts_weights, dim=-1, temperature=5e4) + 1
            cur_pts = torch.clamp(cur_pts, max=self.window_size)

            cur_pts_idx = 0
            for i in range(B):
                last_pt = pts[i][-1] if len(pts[i]) > 0 else self.initial_size
                if last_pt >= self.context_size:
                    continue

                cur_pt = cur_pts[cur_pts_idx]
                seq_end = torch.clamp(cur_pt + last_pt, max=self.context_size)

                pts[i].append(seq_end)

                last_pt_item = last_pt.long() if torch.is_tensor(last_pt) else last_pt

                cur_masked_slice = x[i, last_pt_item: seq_end.long()]

                mask_len = seq_end.long().item() - last_pt_item
                mask_ids = torch.arange(0, mask_len).to(
                    cur_masked_slice.device)
                mask = 1 - torch.sigmoid(mask_ids - cur_pt)

                cur_masked_slice = mask * cur_masked_slice

                x_slices[i].append(cur_masked_slice)

                cur_pts_idx += 1

        pts = [pt_set[:-1] for pt_set in pts]

        return pts, x_slices
