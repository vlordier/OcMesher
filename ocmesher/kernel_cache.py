"""
Optimization: Cache kernel method availability to avoid repeated getattr calls.

The Rust backend checks for evaluate_batch_torch, evaluate_batch, etc. on
every SDF evaluation. We can cache which methods are available for each
kernel to eliminate this overhead.
"""
from typing import Any, Callable, Optional
import functools


class CachedKernel:
    """Wrapper that caches kernel method availability."""
    
    __slots__ = ('_kernel', '_has_eval_batch_torch', '_has_eval_batch', '_has_evaluate_many_torch')
    
    def __init__(self, kernel: Callable):
        self._kernel = kernel
        # Cache method existence checks
        self._has_eval_batch_torch = hasattr(kernel, 'evaluate_batch_torch')
        self._has_eval_batch = hasattr(kernel, 'evaluate_batch')
        self._has_evaluate_many_torch = hasattr(kernel, 'evaluate_many_torch')
    
    def __call__(self, *args, **kwargs):
        """Forward __call__ to wrapped kernel."""
        return self._kernel(*args, **kwargs)
    
    def __getattr__(self, name: str):
        """Forward attribute access to wrapped kernel."""
        return getattr(self._kernel, name)
    
    @property
    def has_evaluate_batch_torch(self) -> bool:
        """Check if kernel has evaluate_batch_torch method (cached)."""
        return self._has_eval_batch_torch
    
    @property
    def has_evaluate_batch(self) -> bool:
        """Check if kernel has evaluate_batch method (cached)."""
        return self._has_eval_batch
    
    @property
    def has_evaluate_many_torch(self) -> bool:
        """Check if kernel has evaluate_many_torch method (cached)."""
        return self._has_evaluate_many_torch
    
    def get_or_call_method(self, method_name: str, *args, **kwargs):
        """Get a method from the kernel and call it."""
        method = getattr(self._kernel, method_name)
        return method(*args, **kwargs)


def wrap_kernels_with_cache(kernels: list[Any]) -> list[CachedKernel]:
    """Wrap kernels to cache method availability checks.
    
    This is typically called once when kernels are first passed to the mesher,
    then the wrapped kernels are reused for all subsequent batches.
    """
    return [CachedKernel(k) if not isinstance(k, CachedKernel) else k for k in kernels]
