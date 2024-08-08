import torch
from torch import nn, Tensor
import torch.nn.functional as F
from domain.rec_enc.config import RecEncConfig


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
        self.n_pred_steps = self.pred_len // self.patch_len

        self.pos_emb = nn.Parameter(torch.randn(
            self.n_pred_steps, self.d_model // 2))
        self.channel_emb = nn.Parameter(
            torch.randn(config.n_channels, self.d_model // 2))

        self.dropout = nn.Dropout(config.dropout)
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)
        # y = ax + b                # 2
        # y = a * sin(bx + c) + d   # 4
        # y = a * tan(bx + c) + d   # 4
        self.trend_params = nn.Linear(
            self.d_model,
            self.n_pred_steps * 2
        )
        with torch.no_grad():
            self.trend_params.weight.data.zero_()
            self.trend_params.bias.data.zero_()
        self.func_params = nn.Linear(
            self.d_model,
            # self.n_pred_steps * (4 + 4)
            self.n_pred_steps * 4
        )
        with torch.no_grad():
            self.func_params.weight.data.normal_(0, 0.02)
            self.func_params.bias.data.fill_(0)

    def forward(self, x: Tensor):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.n_pred_steps

        # B, L, C -> B, C, L -> B * C, N, W
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = F.relu(xd)

        # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d
        enc_out = self.gru(enc_in)[1]
        # enc_out = enc_out.repeat(1, 1, M).view(1, -1, self.d_model)

        # dec_in = torch.cat([
        #     # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
        #     self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1),
        #     # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        #     self.channel_emb.unsqueeze(1).repeat(B, M, 1)
        # ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        # dec_out = self.pred_gru(dec_in, enc_out)[0]  # B * C * M, 1, d
        params = self.trend_params(enc_out[0]).reshape(B * C * M, -1)
        ids = torch.arange(0, self.patch_len).unsqueeze(0).repeat(B * C * M, 1).to(params.device)
        a, b = params[:, :1], params[:, 1:2]
        data = a * ids + b
        params = self.func_params(enc_out[0]).reshape(B * C * M, -1)
        a, b, c, d = params[:, :1], params[:, 1:2], params[:, 2:3], params[:, 3:4]
        sin_data = a * torch.sin(b * ids + c) + d
        data = data + sin_data
        # a, b, c, d = params[:, 4:5], params[:, 5:6], params[:, 6:7], params[:, 7:]
        # tan_data = a * torch.tan(b * ids + c) + d
        # data = data + tan_data
        data = data.reshape(B, C, -1).transpose(-1, -2)
        
        data = data + seq_last

        return data

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1)  # B, C, H

        y = y + seq_last

        return y
