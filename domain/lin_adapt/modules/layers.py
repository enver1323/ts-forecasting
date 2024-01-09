from torch import nn, Tensor
from domain._common.modules.normalization import AdaptiveNorm


class ComponentLinAdapt(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int
    ):
        super(ComponentLinAdapt, self).__init__()

        self.lin = nn.Linear(seq_len, pred_len)
        self.norm = AdaptiveNorm(seq_len)

    def forward(self, x: Tensor) -> Tensor:
        # x = self.norm(x, 'norm')
        x = self.lin(x)
        # x = self.norm(x, 'denorm')

        return x