"""
Reproducibility utilities for ensuring consistent results across runs.

This module provides functions to set random seeds and enable deterministic
behavior across PyTorch, NumPy, and Python's random module.
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed (int): Random seed value to use. Default is 42.
        deterministic (bool): Whether to enable deterministic behavior.
                             Note: This may impact performance.
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic behavior
    if deterministic:
        # Make PyTorch operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Enable deterministic algorithms in PyTorch (if available)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as e:
            print(f"Warning: Could not enable deterministic algorithms: {e}")
    
    print(f"Random seeds set to {seed} with deterministic={deterministic}")


def set_reproducible_environment(seed: int = 42, deterministic: bool = True) -> None:
    """
    Comprehensive function to set up a reproducible environment.
    
    This function sets seeds, enables deterministic behavior, and configures
    the environment for maximum reproducibility.
    
    Args:
        seed (int): Random seed value to use. Default is 42.
        deterministic (bool): Whether to enable deterministic behavior.
    """
    # Set all random seeds
    set_seed(seed, deterministic)
    
    # Additional environment configurations
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("Reproducible environment configured successfully")


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_preference (str): Device preference. Options: 'auto', 'cpu', 'cuda'.
        
    Returns:
        torch.device: The selected device.
    """
    if device_preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_preference == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    return device


def setup_mixed_precision() -> Optional[torch.cuda.amp.GradScaler]:
    """
    Setup mixed precision training for better performance.
    
    Returns:
        torch.cuda.amp.GradScaler or None: GradScaler for mixed precision,
                                          or None if CUDA not available.
    """
    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("Mixed precision training enabled")
        return scaler
    else:
        print("Mixed precision not available (CUDA required)")
        return None


def log_system_info() -> None:
    """
    Log system information for reproducibility documentation.
    """
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("==========================")


if __name__ == "__main__":
    # Example usage
    set_reproducible_environment(seed=42, deterministic=True)
    device = get_device("auto")
    scaler = setup_mixed_precision()
    log_system_info()
