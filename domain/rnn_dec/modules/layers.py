import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

from domain.rnn_dec.config import RNNDecConfig


class StatsPred(nn.Module):
    def __init__(self, config: RNNDecConfig.ModelConfig):
        super(StatsPred, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_channels = config.n_channels
        self.patch_len = config.patch_len
        self.d_model = config.d_model

        self.date_encode = nn.Linear(
            config.n_date_channels * self.patch_len,
            self.d_model
        )
        self.in_embed = nn.Linear(self.patch_len, self.d_model)
        self.in_to_hid = nn.GRU(
            input_size=2 * self.d_model,
            hidden_size=2 * self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.epsilon = 1e-8

        self.pos_emb = nn.Parameter(torch.randn(
            self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(
            torch.randn(self.n_channels, self.d_model // 2))
        self.dec_hid = nn.GRU(
            input_size=2 * self.d_model,
            hidden_size=2 * self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.enc_mean = nn.Linear(2 * self.d_model, 1)
        self.enc_var = nn.Linear(2 * self.d_model, 1)
        self.dec_mean = nn.Linear(2 * self.d_model, 1)
        self.dec_var = nn.Linear(2 * self.d_model, 1)

    def embed_date(self, x: Tensor):
        x = rearrange(x, 'b (n w) c -> b n (w c)', w=self.patch_len)
        x = self.date_encode(x).repeat(self.n_channels, 1, 1)
        return x

    def encode(self, x: Tensor, date: Tensor) -> Tensor:
        # Input embedding
        x = rearrange(x, 'b (n w) c -> (b c) n w', w=self.patch_len)
        x = self.in_embed(x)
        x = F.relu(x)

        # Date embedding
        date = self.embed_date(date)
        x = torch.concatenate([x, date], dim=-1)

        stats, hid = self.in_to_hid(x)
        stats = self.dropout(stats)

        mean = self.enc_mean(stats)
        std = self.enc_var(stats)
        mean = rearrange(mean, '(b c) n s -> s b c n', c=self.n_channels)[0]
        std = rearrange(std, '(b c) n s -> s b c n', c=self.n_channels)[0]

        return (mean, std), hid

    def decode(self, hidden: Tensor, date: Tensor):
        B = date.shape[0]
        date = self.embed_date(date)
        N = date.shape[-2]

        channel_emb = self.channel_emb.unsqueeze(1)\
            .repeat(B, N, 1)
        pos_emb = self.pos_emb.unsqueeze(0)\
            .repeat(B * self.n_channels, 1, 1)

        stats = torch.concatenate([
            pos_emb, channel_emb, date
        ], dim=-1).flatten(0, 1).unsqueeze(-2)

        hidden = hidden.repeat(1, 1, N)
        hidden = rearrange(hidden, 'r bc (n d) -> r (bc n) d', n=N)

        stats = self.dec_hid(stats, hidden)[0]
        stats = self.dropout(stats)

        mean = self.enc_mean(stats)
        std = self.enc_var(stats)

        mean = rearrange(mean, '(b c n) 1 s -> 1 s b c n',
                         c=self.n_channels, n=N)[0][0]
        std = rearrange(std, '(b c n) 1 s -> 1 s b c n',
                        c=self.n_channels, n=N)[0][0]

        return mean, std