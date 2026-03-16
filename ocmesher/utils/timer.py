# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Simple wall-clock timer with memory reporting."""

import logging
import os
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Self

import psutil

logger = logging.getLogger(__name__)

__all__ = ["Timer"]


class Timer:
    """Context manager that measures wall-clock duration and reports memory usage.

    Usage::

        with Timer("my step"):
            do_work()
        # logs: [my step] finished in 0:00:01.234 with memory usage 0.5 GB
    """

    __slots__ = ("disable_timer", "duration", "end", "name", "start")

    disable_timer: bool
    duration: timedelta
    end: datetime
    name: str
    start: datetime

    def __init__(self, desc: str, disable_timer: bool = False) -> None:
        """Create a timer labelled *desc*."""
        self.disable_timer = disable_timer
        if self.disable_timer:
            return
        self.name = f"[{desc}]"

    def __enter__(self) -> Self:
        """Record the start time."""
        if self.disable_timer:
            return self
        self.start = datetime.now(tz=UTC)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Log elapsed time and memory on success, or the exception type on failure."""
        if self.disable_timer:
            return
        self.end = datetime.now(tz=UTC)
        self.duration = self.end - self.start  # timedelta
        if exc_type is None:
            try:
                process = psutil.Process(os.getpid())
                mem_gb = process.memory_info().rss / 1024**3
                logger.info("%s finished in %s with memory usage %.2f GB", self.name, self.duration, mem_gb)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.info("%s finished in %s", self.name, self.duration)
        else:
            logger.warning("%s failed with %s: %s", self.name, exc_type.__name__, _exc_val)
