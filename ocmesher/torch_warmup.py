"""
Optimization: Warm up torch device on first use to avoid JIT overhead.

The first torch operation on a device can be slower due to JIT compilation
and device initialization. This module pre-warms the device to get representative
timing for subsequent operations.
"""
import torch
import numpy as np
from typing import Optional


def warmup_torch_device(device: str = "cpu", num_test_tensors: int = 5) -> float:
    """
    Warm up a torch device by running some dummy operations.
    
    Args:
        device: Device name (cpu, cuda, mps)
        num_test_tensors: Number of test operations to run
    
    Returns:
        Estimated device overhead in seconds (useful for diagnostics)
    
    Note:
        This is called automatically by RustOcMesher on first kernel
        evaluation to eliminate JIT overhead from user-visible runtime.
    """
    import time
    
    if device not in ("cpu", "cuda", "mps"):
        return 0.0
    
    try:
        # Check device availability
        if device == "cuda":
            if not torch.cuda.is_available():
                return 0.0
        elif device == "mps":
            if not hasattr(torch.backends.mps, "is_available") or not torch.backends.mps.is_available():
                return 0.0
        
        t0 = time.perf_counter()
        
        # Quick warm-up: allocate/compute/move tensors
        for _ in range(num_test_tensors):
            x = torch.randn(100, requires_grad=False, device=device)
            y = torch.sin(x)
            z = y.sum()
            if device in ("cuda", "mps"):
                torch.cuda.synchronize() if device == "cuda" else None
        
        elapsed = time.perf_counter() - t0
        return elapsed
    except Exception:
        # On any error, return 0 and continue
        return 0.0
