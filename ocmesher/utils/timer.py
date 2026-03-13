# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Simple wall-clock timer with memory reporting."""

import os
from datetime import datetime, timezone

import psutil


class Timer:
    """Context-manager that logs elapsed time and memory usage."""

    def __init__(self, desc, *, disable_timer=False):
        """Create a timer labelled *desc*."""
        self.disable_timer = disable_timer
        if self.disable_timer:
            return
        self.name = f"[{desc}]"

    def __enter__(self):
        """Record the start time."""
        if self.disable_timer:
            return
        self.start = datetime.now(tz=timezone.utc)

    def __exit__(self, exc_type, _exc_val, _traceback):
        """Print elapsed time and memory on success, or the exception type on failure."""
        if self.disable_timer:
            return
        self.end = datetime.now(tz=timezone.utc)
        self.duration = self.end - self.start  # timedelta
        if exc_type is None:
            process = psutil.Process(os.getpid())
            print(
                f"{self.name} finished in {self.duration!s} with memory usage {process.memory_info().rss / 1024**3} GB"
            )
        else:
            print(f"{self.name} failed with {exc_type}")
