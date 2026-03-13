# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from types import TracebackType


class Timer:
    def __init__(self, desc: str, disable_timer: bool = False) -> None:
        self.disable_timer = disable_timer
        if self.disable_timer:
            return
        self.name = f"[{desc}]"
        self.start: datetime
        self.end: datetime
        self.duration: timedelta

    def __enter__(self) -> Timer:
        if self.disable_timer:
            return self
        self.start = datetime.now()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.disable_timer:
            return
        self.end = datetime.now()
        self.duration = self.end - self.start  # timedelta
        if exc_type is None:
            process = psutil.Process(os.getpid())
            print(
                f"{self.name} finished in {self.duration!s} with memory usage {process.memory_info().rss / 1024**3} GB"
            )
        else:
            print(f"{self.name} failed with {exc_type}")
