# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Context manager for timing code blocks with memory reporting."""

from __future__ import annotations

import os
import time

import psutil


class Timer:
    """Context manager that measures elapsed time and reports memory usage."""

    def __init__(self, desc: str, disable_timer: bool = False) -> None:
        self.disable_timer = disable_timer
        if not self.disable_timer:
            self.name = f"[{desc}]"
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        if not self.disable_timer:
            self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        traceback: object,
    ) -> None:
        if self.disable_timer:
            return
        self.elapsed = time.perf_counter() - self._start
        if exc_type is None:
            process = psutil.Process(os.getpid())
            mem_gb = process.memory_info().rss / (1024**3)
            print(
                f"{self.name} finished in {self.elapsed:.6f}s"
                f" with memory usage {mem_gb:.2f} GB",
            )
        else:
            print(f"{self.name} failed with {exc_type}")
