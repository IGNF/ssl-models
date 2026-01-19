"""Sampler module."""

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

        # total_size == num_samples (NOT multiplied)
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=g,
        )

        # optional slicing for decorrelation, but NOT required
        return iter(indices.tolist())

    def __len__(self):
        # MUST match number of yielded indices
        return self.num_samples
