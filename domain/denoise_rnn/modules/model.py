import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from domain.denoise_rnn.config import DenoiseRNNConfig
from domain.denoise_rnn.modules.layers import MGRU, GLGRU, RWKV, IRNN


class DenoiseRNN(nn.Module):
    def __init__(self, config: DenoiseRNNConfig.ModelConfig):
        super(DenoiseRNN, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_channels = config.n_channels
        self.patch_len = config.patch_len
        self.d_model = config.d_model

        self.n_seq_steps = self.seq_len // self.patch_len
        self.n_pred_steps = self.pred_len // self.patch_len

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        # n_heads = 4
        # self.rwkv = RWKV(
        #     hidden_size=self.d_model,
        #     n_heads=n_heads,
        #     dropout=config.dropout,
        # )

        self.pred_gru = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        # self.pred_rwkv = RWKV(
        #     hidden_size=self.d_model,
        #     n_heads=n_heads,
        #     dropout=config.dropout
        # )
        self.pred_gru = nn.Linear(
            self.d_model, self.n_pred_steps * self.d_model)

        self.pred_lin = nn.Linear(self.d_model, self.d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(self.n_pred_steps, self.d_model // 2))
        self.channel_emb = nn.Parameter(
            torch.randn(self.n_channels, self.d_model // 2))
        self.emb = nn.Parameter(
            torch.randn(self.n_channels * self.n_pred_steps, self.d_model))

        # self.hid_proj = nn.Linear(self.d_model ** 2 // n_heads, self.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x: Tensor):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len

        xw = rearrange(x, 'b (n w) c -> (b c) n w', n=N, w=self.patch_len)
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        enc_out = self.gru(enc_in)[1][0]
        # enc_out = enc_out.repeat(1, 1, M)
        # enc_out = self.rwkv(enc_in)[1]
        # enc_out = rearrange(enc_out, 'b d n m -> b (d n m)')
        # enc_out = self.hid_proj(enc_out).repeat(1, 1, M)
        # enc_out = rearrange(enc_out, 'r bc (m d) -> r (bc m) d', m=M)
        # enc_out = rearrange(enc_out, '1 bc (m d) -> (bc m) 1 d', m=M)

        # dec_in = torch.cat([
        #     # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
        #     self.pos_emb[:M, :].unsqueeze(0).repeat(B*C, 1, 1),
        #     # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        #     self.channel_emb.unsqueeze(1).repeat(B, M, 1)
        # ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        # dec_in = self.emb.repeat(B, 1).unsqueeze(-2)
        # dec_out = self.pred_gru(dec_in, enc_out)[0]  # B * C * M, 1, d
        # dec_out = self.pred_lin(torch.concatenate((dec_in, enc_out), dim=-1))  # B * C * M, 1, d
        dec_out = self.pred_gru(enc_out)
        dec_out = rearrange(dec_out, 'b (m d) -> (b m) d', m=M).unsqueeze(-2)

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = rearrange(yw, '(b c m) 1 w -> b (m w) c', b=B, c=C, m=M)  # B, H, C

        y = y + seq_last

        return y
