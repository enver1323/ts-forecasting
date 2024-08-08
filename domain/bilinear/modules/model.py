import torch
from torch import nn, Tensor
import torch.nn.functional as F

from domain.bilinear.config import BiLinearConfig


class BiLinear(nn.Module):
    def __init__(self, config: BiLinearConfig.ModelConfig):
        super(BiLinear, self).__init__()

        # self.patch_embed = nn.Linear(config.patch_len, config.d_model)
        self.patch_len = config.patch_len
        self.n_x_patches = config.seq_len // config.patch_len
        self.n_y_patches = config.pred_len // config.patch_len

        self.embed = nn.Linear(config.patch_len, config.d_model)
        self.bilin = nn.Bilinear(
            config.d_model, config.d_model, config.d_model)
        self.unembed = nn.Linear(
            config.d_model, config.pred_len)
    
    def step(self, x: Tensor, hid: Tensor):
        # TRY torch.softmax(self.bilin(x1, x2)) @ hid
        hid = self.bilin(x, hid)
        hid = F.tanh(hid)
        return hid


    def forward(self, x: Tensor):
        batch_size, _, n_channels = x.shape
        # seq_last = x[:, -1:, :].detach()
        # x = (x - seq_last)
        x = x.permute(0, 2, 1)

        x = self.embed(
            x.reshape(-1, self.n_x_patches, self.patch_len)
        )
        x = F.relu(x)

        hid = self.step(x[:, 1, :], x[:, 0, :])
        for i in range(1, self.n_x_patches):
            hid = self.step(x[:, i, :], hid)

        x = self.unembed(hid)
        x = x.reshape(batch_size, n_channels, -1)

        x = x.permute(0, 2, 1)
        # x = x + seq_last

        return x
