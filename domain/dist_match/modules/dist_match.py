from torch import nn, Tensor
import torch
from torch.nn import functional as F
from domain.dist_match.config import DistMatchConfig
from typing import Tuple


class DistMatch(nn.Module):
    def __init__(self, config: DistMatchConfig.ModelConfig):
        super(DistMatch, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_channels = config.n_channels
        self.patch_len = config.patch_len
        self.d_model = config.d_model

        self.patch = nn.Linear(self.patch_len, self.d_model)
        self.transform = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.depatch = nn.Linear(self.d_model, self.patch_len)

    def encode(self, x: Tensor) -> Tensor:
        x = x.transpose(-2, -1)
        orig_shape = list(x.shape)

        x = x.reshape(*orig_shape[:-1], -1, self.patch_len)
        x = self.patch(x)
        x = F.gelu(x)

        x = x @ self.transform
        x = F.gelu(x)

        x = self.depatch(x)
        x = x.reshape(orig_shape)
        x = x.transpose(-2, -1)

        return x

    def decode(self, x: Tensor) -> Tensor:
        x = x.transpose(-2, -1)
        orig_shape = list(x.shape)
        x = x.reshape(*orig_shape[:-1], -1, self.patch_len)
        x = self.patch(x)
        x = F.gelu(x)

        x = x @ self.transform.inverse()
        x = F.gelu(x)

        x = self.depatch(x)
        x = x.reshape(orig_shape)
        x = x.transpose(-2, -1)

        return x
