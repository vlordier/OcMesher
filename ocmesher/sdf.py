# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Batched SDF evaluation with optional bounds clamping."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .types import SDF_BATCH_SIZE


def evaluate_sdfs(
    kernels: list[Callable],
    positions: NDArray,
    *,
    sdf_dtype: type = np.float32,
    bounds_min: NDArray | None = None,
    bounds_max: NDArray | None = None,
) -> NDArray:
    """Evaluate *kernels* at *positions* in batches, returning an (N, K) array.

    Args:
        kernels: List of SDF callables; each accepts (N, 3) and returns (N,).
        positions: (N, 3) array of query points.
        sdf_dtype: Output dtype (default ``np.float32``).
        bounds_min: If provided together with *bounds_max*, points outside the
            axis-aligned box ``[bounds_min, bounds_max]`` are clamped to
            SDF = 1 (positive / outside).
        bounds_max: Upper bound (see *bounds_min*).

    Returns:
        (N, K) SDF values where K = len(kernels), dtype *sdf_dtype*.
    """
    n = len(positions)
    if n == 0:
        return np.zeros((0, len(kernels)), dtype=sdf_dtype)

    clamp = bounds_min is not None and bounds_max is not None
    results: list[NDArray] = []

    for start in range(0, n, SDF_BATCH_SIZE):
        batch = positions[start : start + SDF_BATCH_SIZE]

        out_of_bounds: NDArray | None = None
        if clamp:
            out_of_bounds = np.any(batch <= bounds_min, axis=1) | np.any(
                batch >= bounds_max, axis=1,
            )

        batch_sdfs: list[NDArray] = []
        for kernel in kernels:
            sdf = kernel(batch)
            if out_of_bounds is not None:
                sdf[out_of_bounds] = 1
            batch_sdfs.append(sdf)

        results.append(np.stack(batch_sdfs, axis=-1).astype(sdf_dtype))

    return np.concatenate(results, axis=0)
