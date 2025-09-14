"""
Example script demonstrating reproducibility features.

This script shows how to use the reproducibility utilities
to ensure consistent results across runs.
"""

import numpy as np
import torch
from reproducibility import set_reproducible_environment, log_system_info


def demonstrate_reproducibility():
    """
    Demonstrate that the reproducibility setup works correctly.
    """
    print("=== Reproducibility Demonstration ===\n")
    
    # Setup reproducible environment
    set_reproducible_environment(seed=42, deterministic=True)
    
    # Log system information
    log_system_info()
    
    print("\n=== Random Number Generation Tests ===")
    
    # Test NumPy reproducibility
    print("NumPy random numbers (seed=42):")
    np.random.seed(42)
    numpy_vals = np.random.rand(5)
    print(f"First run:  {numpy_vals}")
    
    np.random.seed(42)
    numpy_vals2 = np.random.rand(5)
    print(f"Second run: {numpy_vals2}")
    print(f"Identical:  {np.allclose(numpy_vals, numpy_vals2)}")
    
    # Test PyTorch reproducibility
    print("\nPyTorch random numbers (seed=42):")
    torch.manual_seed(42)
    torch_vals = torch.rand(5)
    print(f"First run:  {torch_vals}")
    
    torch.manual_seed(42)
    torch_vals2 = torch.rand(5)
    print(f"Second run: {torch_vals2}")
    print(f"Identical:  {torch.allclose(torch_vals, torch_vals2)}")
    
    # Test CUDA reproducibility (if available)
    if torch.cuda.is_available():
        print("\nCUDA random numbers (seed=42):")
        torch.cuda.manual_seed(42)
        cuda_vals = torch.rand(5, device='cuda')
        print(f"First run:  {cuda_vals}")
        
        torch.cuda.manual_seed(42)
        cuda_vals2 = torch.rand(5, device='cuda')
        print(f"Second run: {cuda_vals2}")
        print(f"Identical:  {torch.allclose(cuda_vals, cuda_vals2)}")
    else:
        print("\nCUDA not available - skipping CUDA reproducibility test")
    
    print("\n=== Deterministic Operations Test ===")
    
    # Test matrix operations
    print("Matrix multiplication (should be deterministic):")
    torch.manual_seed(42)
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    result1 = torch.mm(a, b)
    print(f"First run:  {result1[0, 0]:.6f}")
    
    torch.manual_seed(42)
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    result2 = torch.mm(a, b)
    print(f"Second run: {result2[0, 0]:.6f}")
    print(f"Identical:  {torch.allclose(result1, result2)}")
    
    print("\n=== Reproducibility Setup Complete ===")
    print("All random number generators are now deterministic.")
    print("You can run this script multiple times to verify consistency.")


if __name__ == "__main__":
    demonstrate_reproducibility()
