"""Probing layers"""

import torch
from torch import nn
from einops import rearrange


class AttentiveProbing(nn.Module):
    def __init__(self, dim, num_classes, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.query = nn.Parameter(torch.randn(inner_dim))

        self.linear = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        out = x.detach()
        out = self.norm(out)

        q = rearrange(self.query, '(H D) -> 1 H 1 D', H=self.heads)
        kv = self.to_kv(out).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'B L (H D) -> B H L D', H=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'B H 1 D -> B (H D)')
        return self.linear(out)


class LinearProbing(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x):
        out = x.detach()
        out = torch.mean(out, dim=1)
        return self.linear(out)
