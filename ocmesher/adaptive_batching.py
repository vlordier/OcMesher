"""
Optimization: Adaptive batch sizing based on kernel complexity estimation.

The profiling shows:
- batch=8k: 24.9s (25.6k SDF calls)
- batch=16k: 18.5s (12.8k SDF calls)  
- batch=32k: 16.2s (6.4k SDF calls)

Larger batches are 35% faster per call due to lower Python/FFI overhead per element.
This module provides auto-tuning to pick optimal batch size based on kernel runtime.
"""
import numpy as np
import torch
import time
from typing import Callable, Optional, Tuple


def estimate_kernel_complexity(
    kernel: Callable,
    n_test_pts: int = 1024,
) -> float:
    """
    Estimate kernel runtime per (N, 3) input batch.
    
    Returns:
        Time in ms per 1000 points.
    """
    xyz = np.random.randn(n_test_pts, 3).astype(np.float64)
    
    t0 = time.perf_counter()
    try:
        result = kernel(xyz)
        elapsed = time.perf_counter() - t0
        # Normalize to ms per 1000 points
        return (elapsed * 1000.0) / n_test_pts
    except Exception:
        # If kernel fails, assume medium complexity
        return 1.0


def adaptive_batch_size(
    kernel_runtime_ms_per_kpts: float,
    base_batch_size: int = 8192,
    max_batch_size: int = 131072,
    min_batch_size: int = 512,
) -> int:
    """
    Compute ideal batch size based on kernel complexity.
    
    Args:
        kernel_runtime_ms_per_kpts: Runtime in ms per 1000 points
        base_batch_size: Default batch size (adjusted via multiplier)
        max_batch_size: Hard upper limit
        min_batch_size: Hard lower limit
    
    Returns:
        Recommended batch size.
    
    Logic:
        - If kernel is very fast (< 0.5 ms/1kpts): use larger batches to reduce FFI overhead
        - If kernel is slow (> 2 ms/1kpts): use smaller batches to avoid GPU memory pressure
        - Otherwise: stay near base_batch_size
    """
    if kernel_runtime_ms_per_kpts < 0.25:
        # Very fast kernel - maximize batch size
        multiplier = 2.0
    elif kernel_runtime_ms_per_kpts < 0.5:
        # Fast kernel
        multiplier = 1.5
    elif kernel_runtime_ms_per_kpts < 2.0:
        # Medium kernel - use base
        multiplier = 1.0
    elif kernel_runtime_ms_per_kpts < 5.0:
        # Slow kernel
        multiplier = 0.7
    else:
        # Very slow kernel - small batches
        multiplier = 0.4
    
    suggested = int(base_batch_size * multiplier)
    return max(min_batch_size, min(max_batch_size, suggested))


def profile_and_recommend_batch_size(
    kernels: list,
    default_batch_size: int = 8192,
) -> Tuple[int, dict]:
    """
    Profile kernels and return recommended batch size + diagnostic dict.
    
    Args:
        kernels: List of SDF kernel callables
        default_batch_size: Fallback batch size
    
    Returns:
        (recommended_batch_size, diagnostics_dict)
    """
    diagnostics = {}
    
    if not kernels:
        return default_batch_size, diagnostics
    
    # Profile each kernel
    complexities = []
    for i, kernel in enumerate(kernels):
        try:
            ms_per_kpts = estimate_kernel_complexity(kernel, n_test_pts=2048)
            complexities.append(ms_per_kpts)
            diagnostics[f"kernel_{i}_complexity_ms_per_1kpts"] = ms_per_kpts
        except Exception as e:
            diagnostics[f"kernel_{i}_error"] = str(e)
            complexities.append(1.0)  # Assume medium complexity on error
    
    # Use worst-case (slowest kernel) to be conservative
    max_complexity = max(complexities) if complexities else 1.0
    diagnostics["max_kernel_complexity_ms_per_1kpts"] = max_complexity
    
    recommended = adaptive_batch_size(max_complexity, base_batch_size=default_batch_size)
    diagnostics["recommended_batch_size"] = recommended
    
    return recommended, diagnostics
