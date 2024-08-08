import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from domain.rnn_dec.config import RNNDecConfig


class RNNDec(nn.Module):
    def __init__(self, config: RNNDecConfig.ModelConfig):
        super(RNNDec, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_channels = config.n_channels
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

        # self.pred_gru = nn.GRU(
        #     input_size=self.d_model,
        #     hidden_size=self.d_model,
        #     num_layers=1,
        #     bias=True,
        #     batch_first=True,
        # )

        self.pos_emb = nn.Parameter(torch.randn(
            self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(
            torch.randn(self.n_channels, self.d_model // 2))

        self.dropout = nn.Dropout(config.dropout)
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)


    def forward(self, x: Tensor):
        # seq_last = x[:, -1:, :].detach()
        # x = x - seq_last

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len

        xw = rearrange(x, 'b (n w) c -> (b c) n w', w=self.patch_len)

        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = F.relu(xd)

        enc_out = self.gru(enc_in)[1][0].repeat(1, 1, M)
        enc_out = rearrange(enc_out, 'r bc (m d) -> r (bc m) d', m=M)

        dec_in = torch.cat([
            # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1),
            # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1)
            # ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d -> B * C * M, 1, d
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d
        dec_out = self.gru(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W

        y = rearrange(yw, '(b c m) 1 w -> b (m w) c', b=B, c=C, m=M)  # B, H, C

        # y = y + seq_last

        return y
