from torch import nn, Tensor
from domain.dist_match.config import DistMatchConfig


class Linear(nn.Module):
    def __init__(self, config: DistMatchConfig.ModelConfig):
        super(Linear, self).__init__()

        self.layer = nn.Linear(config.seq_len, config.pred_len)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(-1, -2)
        x = self.layer(x)
        x = x.transpose(-1, -2)
        return x
