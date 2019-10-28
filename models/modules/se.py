import torch
import torch.nn as nn
from sparse_layers.sparse_layers import SparseLinear
from .activations import Swish, HardSwish, HardSigmoid


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.ratio = ratio
        self.relu = nn.ReLU(True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.global_pool(x).flatten(1, -1)
        mask = self.transform(x_avg)
        return x * mask.unsqueeze(-1).unsqueeze(-1)


class SESwishBlock(nn.Module):
    """ squeeze-excite block for MBConv """

    def __init__(self, in_channels, out_channels=None, interm_channels=None, ratio=None, hard_act=False, sparsity=None):
        super(SESwishBlock, self).__init__()
        assert not (interm_channels is None and ratio is None)
        interm_channels = interm_channels or in_channels // ratio
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.ratio = ratio
        self.activation = HardSwish() if hard_act else Swish(),
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if sparsity is not None:
            linear1 = SparseLinear(in_channels, interm_channels, nonzero_frac=sparsity)
            linear2 = SparseLinear(interm_channels, out_channels, nonzero_frac=sparsity)
        else:
            linear1 = nn.Linear(in_channels, interm_channels)
            linear2 = nn.Linear(interm_channels, out_channels)
        
        self.transform = nn.Sequential(
            linear1,
            HardSwish() if hard_act else Swish(),
            linear2,
            HardSigmoid() if hard_act else nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.global_pool(x).flatten(1, -1)
        mask = self.transform(x_avg)
        return x * mask.unsqueeze(-1).unsqueeze(-1)
