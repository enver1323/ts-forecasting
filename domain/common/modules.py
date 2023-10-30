import torch
from torch import nn, Tensor, Size
from torch.nn import functional as F
from typing import Tuple, Type, Sequence, Optional, Union, List
import numpy as np
import math


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=stride, padding=0
        )

    def forward(self, data: Tensor) -> Tensor:
        front = data[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        end = data[:, -1:, :].repeat(1, math.ceil(self.kernel_size / 2) - 1, 1)

        x = torch.cat([front, data, end], dim=-2)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, decomposer: nn.Module):
        super(SeriesDecomposition, self).__init__()

        self.decomposer = decomposer

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        trend = self.decomposer(data)
        seasonality = data - trend
        return trend, seasonality


def get_channelwise_modules(n_channels: int, module: Type[nn.Module], args: Sequence) -> nn.ModuleList:
    return nn.ModuleList([
        module(*args)
        for i in range(n_channels)
    ])


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
    
class AdaptiveNorm(nn.Module):
    def __init__(self, n_params: int, eps: float = 1e-5):
        super(AdaptiveNorm, self).__init__()
        self.eps = eps
        self.mean_adapt = nn.Linear(n_params, 1)
        self.std_adapt = nn.Linear(n_params, 1)

        self.__init_weights_uniform(self.mean_adapt)
        self.__init_weights_uniform(self.std_adapt)

    def __init_weights_uniform(self, module: nn.Linear):
        n = module.in_features
        y = 1.0/np.sqrt(n)
        module.weight.data.uniform_(-y, y)
        module.bias.data.fill_(0)

    def forward(self, x: Tensor, mode:str ='norm', dims: Union[int, Size] = -1):
        if mode == 'norm':
            x = self.norm(x, dims)
        elif mode == 'denorm':
            x = self.denorm(x)

        return x
    
    def norm(self, x: Tensor, dims: Union[int, Size]) -> Tensor:
        mean = torch.mean(x, dim=dims, keepdim=True).detach()
        std = torch.sqrt(
            torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

        x = (x - mean) / std
        
        scale = 1
        for dim in list(mean.shape):
            scale /= dim
        self.mean = self.mean_adapt(x) * scale + mean
        self.std = self.std_adapt(x) * scale + std

        return x
    
    def denorm(self, x: Tensor) -> Tensor:
        x = x * self.std + self.mean
        return x

        

class DishTS(nn.Module):
    def __init__(self, init_type: str, n_series: int, seq_len: int, activate: bool = True):
        super().__init__()
        init = init_type #'standard', 'avg' or 'uniform'
        n_series = n_series # number of series
        lookback = seq_len # lookback length
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(n_series, lookback, 2)/lookback)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(torch.ones(n_series, lookback, 2)/lookback+torch.rand(n_series, lookback, 2)/lookback)
        self.gamma, self.beta = nn.Parameter(torch.ones(n_series)), nn.Parameter(torch.zeros(n_series))
        self.activate = activate

    def forward(self, batch_x, mode='forward', dec_inp=None):
        if mode == 'forward':
            # batch_x: B*L*D || dec_inp: B*?*D (for xxformers)
            self.preget(batch_x)
            batch_x = self.forward_process(batch_x)
            dec_inp = None if dec_inp is None else self.forward_process(dec_inp)
            return batch_x, dec_inp
        elif mode == 'inverse':
            # batch_x: B*H*D (forecasts)
            batch_y = self.inverse_process(batch_x)
            return batch_y

    def preget(self, batch_x):
        x_transpose = batch_x.permute(2,0,1) 
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1,2,0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil, self.phih = theta[:,:1,:], theta[:,1:,:] 
        self.xil = torch.sum(torch.pow(batch_x - self.phil,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)
        self.xih = torch.sum(torch.pow(batch_x - self.phih,2), axis=1, keepdim=True) / (batch_x.shape[1]-1)

    def forward_process(self, batch_input):
        #print(batch_input.shape, self.phil.shape, self.xih.shape)
        temp = (batch_input - self.phil)/torch.sqrt(self.xil + 1e-8)
        rst = temp.mul(self.gamma) + self.beta
        return rst
    
    def inverse_process(self, batch_input):
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih