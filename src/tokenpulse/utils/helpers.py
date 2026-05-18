"""
Utility helper functions.
"""

from typing import Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def convert_to_numpy(
    data: Union[np.ndarray, "torch.Tensor"]
) -> np.ndarray:
    """
    Convert input to NumPy array.

    Args:
        data: NumPy array or PyTorch tensor

    Returns:
        NumPy array
    """
    if HAS_TORCH and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def convert_to_torch(
    data: Union[np.ndarray, "torch.Tensor"],
    device: str = None
) -> "torch.Tensor":
    """
    Convert input to PyTorch tensor.

    Args:
        data: NumPy array or PyTorch tensor
        device: Target device (default: auto)

    Returns:
        PyTorch tensor
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this function")

    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.from_numpy(np.asarray(data))

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def get_device() -> str:
    """
    Get the best available device.

    Returns:
        Device string ("cuda" or "cpu")
    """
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
