import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from domain.seg_rnn.config import SegRNNConfig


class SegRNN(nn.Module):
    def __init__(self, config: SegRNNConfig.ModelConfig):
        super(SegRNN, self).__init__()

        # get parameters
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_channels = config.n_channels
        self.d_model = config.d_model
        self.dropout = config.dropout

        self.patch_len = config.patch_len
        self.n_x_patches = self.seq_len//self.patch_len
        self.n_y_patches = self.pred_len // self.patch_len

        self.patch_embed = nn.Sequential(
            nn.Linear(self.patch_len, self.d_model),
            nn.ReLU()
        )

        self.rnn = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.rnn_pred = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
            bidirectional=False
        )
        self.pos_emb = nn.Parameter(
            torch.randn(self.n_y_patches, self.d_model // 2)
        )
        self.channel_emb = nn.Parameter(
            torch.randn(self.n_channels, self.d_model // 2)
        )
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.patch_len)
        )

    def forward(self, x: Tensor):
        batch_size = x.size(0)

        seq_last = x[:, -1:, :].detach()
        x = (x - seq_last)
        x = x.permute(0, 2, 1)

        x = self.patch_embed(
            x.reshape(-1, self.n_x_patches, self.patch_len)
        )

        _, hn = self.rnn(x)  # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.n_channels, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.n_y_patches, 1)
        ], dim=-1).reshape(-1, 1, self.d_model).repeat(batch_size, 1, 1)

        _, hy = self.rnn_pred(pos_emb, hn.repeat(1, 1, self.n_y_patches)
                              .reshape(1, -1, self.d_model))  # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).reshape(-1, self.n_channels, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1)
        y = y + seq_last

        return y
