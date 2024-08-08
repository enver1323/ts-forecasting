from typing import Optional, Tuple, Sequence

import torch
import warnings
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange


class SingleStateRNNBase(nn.Module):
    def __init__(
        self,
        cell: nn.Module,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
    ):
        super(SingleStateRNNBase, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.cell = cell

    def _index(self, x: Tensor, idx, dim: int = -1):
        slices = [slice(None)] * x.ndim
        slices[dim] = idx
        return x[slices]

    def forward(self, x: Tensor, o_h: Optional[Tensor] = None):
        h = o_h if o_h is not None else torch.zeros((self.hidden_size))

        if self.batch_first:
            x = x.transpose(0, 1)
            h = o_h if o_h is not None else torch.zeros(
                (x.size(1), self.hidden_size)
            )

        h = h.to(x.device)
        hs_shape = list(x.shape)
        hs_shape[-1] = self.hidden_size
        hs = torch.empty(hs_shape).to(x.device)

        for i, _x in enumerate(x):
            h = self.cell(_x, h)
            hs[i] = h

        if self.batch_first:
            hs = hs.transpose(0, 1)

        return hs, h.unsqueeze(0)


class MGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super(MGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, 2 * self.hidden_size, bias=bias)
        self.U_f = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        self.U_h = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        x_f, x_h = torch.split(self.W(x), self.hidden_size, dim=-1)
        h_f = self.U_f(h)
        f = torch.sigmoid(x_f + h_f)
        h_h = torch.tanh(x_h + self.U_h(f * h))
        h = (1 - f) * h + f * h_h
        return h


class MGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super(MGRU, self).__init__()

        self.ssrnn = SingleStateRNNBase(
            MGRUCell(input_size, hidden_size, bias),
            hidden_size,
            num_layers,
            batch_first,
        )

    def forward(self, x: Tensor, o_h: Optional[Tensor] = None):
        return self.ssrnn(x, o_h)


class GLGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super(GLGRUCell, self).__init__()

        self.hidden_size = hidden_size

        self.X = nn.Linear(input_size, 4 * self.hidden_size, bias=bias)
        self.H = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=bias)
        self.G_H = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=bias)

        self.H_d = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=bias)
        self.G = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=bias)

    def forward(self, x: Tensor, h: Tensor, g: Tensor) -> Tuple[Tensor, Tensor]:
        x_z, x_r, x_h, x_u = torch.split(self.X(x), self.hidden_size, dim=-1)

        h_z, h_r, h_u = torch.split(self.H(h), self.hidden_size, dim=-1)
        g_z, g_r, g_u = torch.split(self.G_H(g), self.hidden_size, dim=-1)
        z = torch.sigmoid(x_z + h_z + g_z)
        r = torch.sigmoid(x_r + h_r + g_r)
        u = torch.sigmoid(x_u + h_u + g_u)
        h_new = (1 - z) * h + z * torch.tanh(x_h + r * h + u * g)

        h_d = h_new - h
        g_z, g_r = torch.split(self.G(g), self.hidden_size, dim=-1)
        hd_z, hd_r, hd_g = torch.split(
            self.H_d(h_d), self.hidden_size, dim=-1
        )
        z = torch.sigmoid(hd_z + g_z)
        r = torch.sigmoid(hd_r + g_r)
        g = (1 - z) * g + z * torch.tanh(hd_g + r * g)

        return h_new, g


class GLGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super(GLGRU, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.cell = GLGRUCell(input_size, hidden_size, bias)

    def forward(self, x: Tensor, o_h: Optional[Tensor] = None, o_g: Optional[Tensor] = None):
        h = o_h if o_h is not None else torch.zeros((self.hidden_size))
        g = o_g if o_g is not None else torch.zeros((self.hidden_size))

        if self.batch_first:
            x = x.transpose(0, 1)
            h = o_h if o_h is not None else torch.zeros(
                (x.size(1), self.hidden_size)
            )
            g = o_g if o_g is not None else torch.zeros(
                (x.size(1), self.hidden_size)
            )

        h = h.to(x.device)
        g = g.to(x.device)

        hs_shape = list(x.shape)
        hs_shape[-1] = self.hidden_size
        hs = torch.empty(hs_shape).to(x.device)
        for i, _x in enumerate(x):
            h, g = self.cell(_x, h, g)
            hs[i] = h

        h = h.unsqueeze(0)
        g = g.unsqueeze(0)

        if self.batch_first:
            hs = hs.transpose(0, 1)

        return hs, (h, g)


class IRNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super(IRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, self.hidden_size, bias=bias)
        self.U = nn.Parameter(torch.randn(self.hidden_size))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        h = torch.tanh(self.W(x) + self.U * h)
        return h


class IRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super(IRNN, self).__init__()

        self.ssrnn = SingleStateRNNBase(
            IRNNCell(input_size, hidden_size, bias),
            hidden_size,
            num_layers,
            batch_first,
        )

    def forward(self, x: Tensor, o_h: Optional[Tensor] = None):
        return self.ssrnn(x, o_h)


class RWKV(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        dropout: float,
        n_layers: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0

        self.head_size = hidden_size // n_heads
        self.n_head = n_heads

        with torch.no_grad():
            ratio_0_to_1 = 0
            ratio_1_to_almost0 = 1  # 1 to ~0
            ddd = torch.ones(1, 1, hidden_size)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            self.time_maa_k = nn.Parameter(
                1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(
                1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            # self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * \
                    (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.key = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=bias)
        # self.gate = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.output = nn.Linear(hidden_size, hidden_size, bias=bias)
        # self.ln_x = nn.GroupNorm(self.n_head, hidden_size, eps=(1e-5)*64)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state: Optional[Tensor] = None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        H, N = self.n_head, self.head_size
        #
        # we divide a block into chunks to speed up computation & save vram.
        # you can try to find the optimal chunk_len for your GPU.
        # avoid going below 128 if you are using bf16 (otherwise time_decay might be less accurate).
        #
        if T % 256 == 0:
            Q = 256
        elif T % 128 == 0:
            Q = 128
        else:
            Q = T
            warnings.warn(f'\n{"#"*80}\n\n{" "*38}Note\nThe GPT-mode forward() should only be called when we are training models.\nNow we are using it for inference for simplicity, which works, but will be very inefficient.\n\n{"#"*80}\n')
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        # xg = x + xx * self.time_maa_g
        r = rearrange(self.receptance(xr), 'b t (h n) -> b h t n',
                      h=H, n=N)  # receptance
        k = rearrange(self.key(xk), 'b t (h n) -> b h n t', h=H, n=N)  # key
        v = rearrange(self.value(xv), 'b t (h n) -> b h t n',
                      h=H, n=N)  # value
        # g = F.silu(self.gate(xg)) # extra gate

        w = torch.exp(-torch.exp(self.time_decay.float()))  # time_decay
        u = self.time_faaaa.float()  # time_first

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(
            Q-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2*Q - 1)
        w = w[:, :, Q-1:].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype)  # the decay matrix
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = state if state is not None else torch.zeros(
            B, H, N, N, device=r.device, dtype=r.dtype)  # state
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype)  # output

        for i in range(T // Q):  # the rwkv-x051a operator
            rr = r[:, :, i*Q:i*Q+Q, :]
            kk = k[:, :, :, i*Q:i*Q+Q]
            vv = v[:, :, i*Q:i*Q+Q, :]
            y[:, :, i*Q:i*Q+Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = ws * state + (kk * wk) @ vv

        # y = y.transpose(1, 2).contiguous().view(B * T, C)
        # y = self.ln_x(y).view(B, T, C) * g
        y = rearrange(y, 'b h t c -> b t (h c)')

        # output projection
        y = self.dropout(self.output(y))
        return y, state