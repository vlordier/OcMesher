# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Compatibility TorchOcMesher entrypoint backed by Rust runtime.

This preserves the historical ``ocmesher.torch_core`` module path expected by
benchmarks and downstream callers.
"""

from __future__ import annotations

from .core import OcMesher
from .rust_backend import make_rust_ocmesher

__all__ = ["TorchOcMesher"]


class TorchOcMesher(OcMesher):
    """Torch-compatible OcMesher wrapper with legacy constructor signature."""

    def __init__(
        self,
        cameras,
        bounds,
        pixels_per_cube: int = 8,
        inv_scale: int = 10,
        min_dist: int = 1,
        memory_limit_mb: int = 1000,
        bisection_iters: int = 15,
        enclosed: bool = True,
        simplify_occluded: bool = True,
        visible_relax_iter: int = 2,
        coarse_count: int = 500000,
        device: str | None = None,
        n_sdf_workers: int = 4,
        use_compile: bool = False,
    ):
        super().__init__(
            cameras,
            bounds,
            pixels_per_cube=pixels_per_cube,
            inv_scale=inv_scale,
            min_dist=min_dist,
            memory_limit_mb=memory_limit_mb,
            bisection_iters=bisection_iters,
            bisection_tol=0.0,
            enclosed=enclosed,
            simplify_occluded=simplify_occluded,
            visible_relax_iter=visible_relax_iter,
            coarse_count=coarse_count,
        )
        self.device = device or "cpu"
        self.n_sdf_workers = n_sdf_workers
        self.use_compile = use_compile

    def __call__(self, kernels):
        if self._backend_mesher is None:
            self._backend_mesher = make_rust_ocmesher(
                self.cameras,
                self.bounds,
                pixels_per_cube=self.pixels_per_cube,
                inv_scale=self.inv_scale,
                min_dist=self.min_dist,
                memory_limit_mb=self.memory_limit_mb,
                bisection_iters=self.bisection_iters,
                bisection_tol=self.bisection_tol,
                enclosed=self.enclosed,
                simplify_occluded=self.simplify_occluded,
                visible_relax_iter=self.visible_relax_iter,
                coarse_count=self.coarse_count,
                device=self.device,
            )
        return self._backend_mesher(kernels)
