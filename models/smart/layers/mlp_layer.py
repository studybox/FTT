
import torch
import torch.nn as nn

from models.smart.utils import weight_init


class MLPLayer(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MLP(nn.Module):
    def __init__(self, n_in, n_out, ng=32):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_in, n_out)
        self.linear2 = nn.Linear(n_out, n_out)
        self.elu = nn.ELU(inplace=True)
        self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
        self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)

        self.init_weights()

    def forward(self, x):
        out = self.linear1(x)
        out = self.elu(out)
        out = self.norm1(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.norm2(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias, 0.1)