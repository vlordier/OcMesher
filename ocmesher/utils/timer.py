# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

"""Simple wall-clock timers and reusable phase accumulation helpers."""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from time import perf_counter_ns
from types import TracebackType
from typing import Self

import psutil

logger = logging.getLogger(__name__)

__all__ = ["PhaseTracker", "Timer"]


class PhaseTracker:
    """Accumulate named phase durations across one pipeline run."""

    __slots__ = ("disable_timer", "label", "phases")

    disable_timer: bool
    label: str
    phases: dict[str, timedelta]

    def __init__(self, label: str, *, disable_timer: bool = False) -> None:
        self.disable_timer = disable_timer
        self.label = label
        self.phases = {}

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        """Measure one phase block and accumulate its duration by *name*."""
        if self.disable_timer:
            yield
            return
        start_ns = perf_counter_ns()
        try:
            yield
        finally:
            elapsed = timedelta(seconds=(perf_counter_ns() - start_ns) / 1_000_000_000)
            self.add(name, elapsed)

    def add(self, name: str, duration: timedelta) -> None:
        """Accumulate an already measured *duration* under *name*."""
        self.phases[name] = self.phases.get(name, timedelta()) + duration

    def snapshot(self) -> dict[str, timedelta]:
        """Return a copy of the accumulated phase durations."""
        return dict(self.phases)

    def snapshot_millis(self) -> dict[str, float]:
        """Return accumulated phase durations in milliseconds."""
        return {name: duration.total_seconds() * 1000.0 for name, duration in self.phases.items()}

    def log_summary(self) -> None:
        """Emit the accumulated phase summary through the shared timer logger."""
        Timer.log_phase_summary(self.label, self.phases, disable_timer=self.disable_timer)


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

    @staticmethod
    def log_phase_summary(label: str, phases: dict[str, timedelta], *, disable_timer: bool = False) -> None:
        """Log a compact ordered summary of named phase durations."""
        if disable_timer:
            return
        total = sum(phases.values(), start=timedelta())
        phase_parts = ", ".join(f"{name}={duration}" for name, duration in phases.items())
        logger.info("[%s] phase summary: %s, total=%s", label, phase_parts, total)
