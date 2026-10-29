"""Budget-matched low-rank and episodic metric adaptation components."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha/rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x,self.a),self.b)*self.scale


class EpisodicProjection(nn.Module):
    def __init__(self, dim=384, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim,hidden), nn.ReLU(), nn.Linear(hidden,dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.normalize(x + .1*self.net(x),dim=-1)


def proto_logits(support, labels, query, scale=10):
    centers = torch.stack([support[labels == c].mean(0) for c in [0,1]])
    return -torch.cdist(query,centers).square()*scale
