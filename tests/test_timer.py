# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Tests for ocmesher.utils.timer module."""

import time

import pytest

from ocmesher.utils.timer import Timer


class TestTimerBasic:
    """Tests for basic Timer functionality."""

    def test_timer_prints_duration(self, capsys):
        with Timer("test op"):
            time.sleep(0.01)
        captured = capsys.readouterr()
        assert "[test op] finished in" in captured.out
        assert "GB" in captured.out

    def test_timer_disabled(self, capsys):
        with Timer("disabled", disable_timer=True):
            time.sleep(0.01)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_timer_records_duration(self):
        with Timer("duration test") as t:
            time.sleep(0.05)
        assert t.duration.total_seconds() >= 0.04

    def test_timer_context_returns_self(self):
        with Timer("self test") as t:
            assert isinstance(t, Timer)

    def test_timer_disabled_returns_self(self):
        with Timer("disabled self", disable_timer=True) as t:
            assert isinstance(t, Timer)


class TestTimerName:
    """Tests for Timer name formatting."""

    def test_name_formatted_with_brackets(self):
        t = Timer("my task")
        assert t.name == "[my task]"

    def test_disabled_has_no_name(self):
        t = Timer("my task", disable_timer=True)
        assert not hasattr(t, "name")


class TestTimerExceptionHandling:
    """Tests for Timer behavior during exceptions."""

    def test_prints_failure_on_exception(self, capsys):
        with pytest.raises(ValueError, match="test error"):
            with Timer("failing op"):
                msg = "test error"
                raise ValueError(msg)
        captured = capsys.readouterr()
        assert "[failing op] failed with" in captured.out
        assert "ValueError" in captured.out

    def test_does_not_suppress_exception(self):
        with pytest.raises(RuntimeError):
            with Timer("error test"):
                msg = "boom"
                raise RuntimeError(msg)

    def test_disabled_timer_does_not_affect_exception(self):
        with pytest.raises(ZeroDivisionError):
            with Timer("disabled error", disable_timer=True):
                _ = 1 / 0


class TestTimerMemoryReporting:
    """Tests for Timer memory usage reporting."""

    def test_memory_output_format(self, capsys):
        with Timer("mem test"):
            pass
        captured = capsys.readouterr()
        # Should contain formatted memory like "0.12 GB"
        assert "GB" in captured.out
        # Memory value should be a float
        parts = captured.out.strip().split()
        gb_idx = parts.index("GB")
        mem_val = float(parts[gb_idx - 1])
        assert mem_val >= 0

    def test_handles_psutil_error_gracefully(self, capsys, monkeypatch):
        import psutil
        original_process = psutil.Process

        def bad_process(pid):
            raise psutil.NoSuchProcess(pid)

        monkeypatch.setattr(psutil, "Process", bad_process)
        with Timer("psutil error test"):
            pass
        captured = capsys.readouterr()
        assert "[psutil error test] finished in" in captured.out
        # Should NOT contain GB since psutil failed
        assert "GB" not in captured.out
