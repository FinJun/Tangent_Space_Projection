"""Neural network architectures."""

from torch import nn


class fcNet(nn.Module):
    """Fully connected network for cost prediction."""

    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


def build_network(input_dim, output_dim, hidden=None):
    """Build fcNet with given architecture."""
    hidden = hidden or []
    arch = [input_dim] + hidden + [output_dim]
    return fcNet(arch)
