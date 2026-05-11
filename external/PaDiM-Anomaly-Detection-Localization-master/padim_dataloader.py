"""Shared DataLoader settings to improve GPU throughput on fast GPUs (e.g. RTX 3090)."""
import os

import torch
from torch.utils.data import DataLoader


def enable_fast_gpu():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # Ampere+ (e.g. RTX 3090): allow TF32 for conv / matmul speedups
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')


def default_num_workers():
    if not torch.cuda.is_available():
        return 0
    return min(8, os.cpu_count() or 4)


def default_batch_size(arch):
    """Larger batches for feature extraction; tune down if OOM."""
    if arch == 'resnet18':
        return 128
    if arch == 'wide_resnet50_2':
        return 64
    raise ValueError(arch)


def make_feature_loader(dataset, arch, batch_size=None, num_workers=None, shuffle=False):
    bs = batch_size if batch_size is not None else default_batch_size(arch)
    nw = default_num_workers() if num_workers is None else num_workers
    pin = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
