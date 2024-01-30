import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Sequence, Type
from domain._common.utils import get_channelwise_modules
from domain._common.modules.normalization import RevIN
from domain._common.modules.decomposition import SeriesDecomposition, MovingAverage
from domain.rec_enc.modules.layers import TrendPointEstimator, SeasonPointEstimator
from domain.rec_enc.config import RecEncConfig
from einops import rearrange


class RecEnc(nn.Module):
    def __init__(self, config: RecEncConfig.ModelConfig):
        super(RecEnc, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.d_model = config.d_model

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)

        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pred_gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(
            self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(
            torch.randn(config.n_channels, self.d_model // 2))

        self.dropout = nn.Dropout(config.dropout)
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x: Tensor):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len

        # B, L, C -> B, C, L -> B * C, N, W
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = F.relu(xd)

        # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d
        enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model)

        dec_in = torch.cat([
            # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1),
            # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        dec_out = self.pred_gru(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1)  # B, C, H

        y = y + seq_last

        return y
