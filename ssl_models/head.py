"""Head layers"""

import torch
from einops import rearrange
from torch import nn


class Head(nn.Module):
    def maybe_detach_features(self, x, ssl_phase):
        if ssl_phase == "probe":
            x = x.detach()
        return x

    def maybe_detach_prediction(self, x, ssl_phase):
        if ssl_phase == "pretrain":
            x = x.detach()
        return x


class AttentiveHead(Head):
    def __init__(self, dim, num_classes, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.query = nn.Parameter(torch.randn(inner_dim))

        self.linear = nn.Linear(inner_dim, num_classes)

    def forward(self, x, ssl_phase):
        out = self.maybe_detach_features(x, ssl_phase)
        out = self.norm(out)

        q = rearrange(self.query, "(H D) -> 1 H 1 D", H=self.heads)
        kv = self.to_kv(out).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, "B L (H D) -> B H L D", H=self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "B H 1 D -> B (H D)")
        out = self.linear(out)
        out = self.maybe_detach_prediction(out, ssl_phase)
        return out


class LinearHead(Head):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, x, ssl_phase):
        out = self.maybe_detach_features(x, ssl_phase)
        out = torch.mean(out, dim=1)
        out = self.linear(out)
        out = self.maybe_detach_prediction(out, ssl_phase)
        return out
