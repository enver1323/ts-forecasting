import torch
from torch import nn, Tensor
from typing import Type, Sequence


def get_channelwise_modules(
    n_channels: int,
    module: nn.Module,
    args: Sequence,
) -> nn.ModuleList:
    return nn.ModuleList([
        module(*args)
        for i in range(n_channels)
    ])


def arange_like(x: Tensor, dim: int = -1) -> Tensor:
    ids_shape = [1] * x.ndim
    ids_shape[dim] = x.shape[dim]
    indices = torch.arange(0, x.shape[dim]).reshape(ids_shape)
    ids_shape = list(x.shape)
    ids_shape[dim] = 1
    indices = indices.repeat(ids_shape)
    return indices


def soft_argmax(x: Tensor, dim: int = -1, keepdim: bool = False, temperature: float = 1.) -> Tensor:
    indices = arange_like(x, dim).to(x.device)
    x = x * temperature
    x = x.softmax(dim)
    argmax = (x * indices).sum(dim=dim, keepdim=keepdim)

    return argmax
