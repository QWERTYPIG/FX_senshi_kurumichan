"""Window normalization shared by nonprice inputs."""

import torch


def zscore_window_(values, available=None):
    """Z-score each channel over time; absent cells and constant channels become zero."""
    if available is None:
        std, mean = torch.std_mean(values, dim=1, correction=0, keepdim=True)
        values -= mean
    else:
        count = available.sum(dim=1, keepdim=True).clamp_min(1)
        values.masked_fill_(~available, 0)
        values -= values.sum(dim=1, keepdim=True) / count
        values.masked_fill_(~available, 0)
        std = (values.square().sum(dim=1, keepdim=True) / count).sqrt()
    values /= torch.where(std > 0, std, 1)
