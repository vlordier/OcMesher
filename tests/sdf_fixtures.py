"""Reusable SDF kernel factories for tests.

Provides named kernel constructors that replace inline lambda expressions,
improving test readability and eliminating E731 (lambda assignment) warnings.
"""

from __future__ import annotations

import numpy as np


def constant_kernel(value: float):
    """Return a kernel that always produces *value* (float32)."""

    def kernel(x):
        return np.full(len(x), value, dtype=np.float32)

    return kernel
