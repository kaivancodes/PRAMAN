"""Device management for Module 2.

Determines optimal compute device (CUDA GPU, Apple MPS, or CPU fallback).
"""

from typing import Optional, Union
import torch


def get_device(requested_device: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    """Resolve compute device based on configuration and hardware availability.

    Args:
        requested_device: 'auto', 'cuda', 'mps', 'cpu', torch.device, or None.

    Returns:
        torch.device instance.
    """
    if isinstance(requested_device, torch.device):
        return requested_device

    if requested_device and str(requested_device).lower() != "auto":
        dev_str = str(requested_device).lower()
        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        if dev_str == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("cpu")
        return torch.device(dev_str)

    # Automatic selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
