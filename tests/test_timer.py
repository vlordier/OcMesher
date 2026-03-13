"""Tests for ocmesher/utils/timer.py — Timer context manager."""

import io
import sys
import time
from unittest.mock import patch

import pytest

from ocmesher.utils.timer import Timer


class TestTimerNormal:
    """Happy-path timer usage."""

    def test_context_manager_prints_on_success(self, capsys):
        with Timer("test_op"):
            pass
        captured = capsys.readouterr()
        assert "[test_op]" in captured.out
        assert "finished in" in captured.out
        assert "memory usage" in captured.out

    def test_duration_recorded(self):
        t = Timer("dur_test")
        with t:
            pass
        assert hasattr(t, "duration")
        assert t.duration.total_seconds() >= 0


class TestTimerDisabled:
    """Timer with disable_timer=True should be a no-op."""

    def test_no_output_when_disabled(self, capsys):
        with Timer("disabled", disable_timer=True):
            pass
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_duration_attribute_when_disabled(self):
        t = Timer("disabled", disable_timer=True)
        with t:
            pass
        assert not hasattr(t, "duration")

    def test_enter_returns_none_when_disabled(self):
        t = Timer("disabled", disable_timer=True)
        result = t.__enter__()
        assert result is None


class TestTimerException:
    """Timer should still run __exit__ on exceptions."""

    def test_prints_failure_message(self, capsys):
        with pytest.raises(ValueError):
            with Timer("failing_op"):
                raise ValueError("boom")
        captured = capsys.readouterr()
        assert "[failing_op]" in captured.out
        assert "failed with" in captured.out
        assert "ValueError" in captured.out

    def test_exception_propagates(self):
        with pytest.raises(ZeroDivisionError):
            with Timer("div_zero"):
                1 / 0

    def test_disabled_timer_propagates_exception(self):
        with pytest.raises(RuntimeError):
            with Timer("disabled_exc", disable_timer=True):
                raise RuntimeError("error")


class TestTimerEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_description(self, capsys):
        with Timer(""):
            pass
        captured = capsys.readouterr()
        assert "[]" in captured.out

    def test_special_chars_in_description(self, capsys):
        with Timer("test [special] <chars> & more"):
            pass
        captured = capsys.readouterr()
        assert "[test [special] <chars> & more]" in captured.out

    def test_nested_timers(self, capsys):
        with Timer("outer"):
            with Timer("inner"):
                pass
        captured = capsys.readouterr()
        assert "[inner]" in captured.out
        assert "[outer]" in captured.out

    def test_name_attribute_set(self):
        t = Timer("my_timer")
        assert t.name == "[my_timer]"

    def test_name_not_set_when_disabled(self):
        t = Timer("disabled", disable_timer=True)
        assert not hasattr(t, "name")

    def test_reuse_timer_object(self, capsys):
        """Using the same Timer object twice should work for each invocation."""
        t = Timer("reuse")
        with t:
            pass
        captured1 = capsys.readouterr()
        assert "[reuse]" in captured1.out

        with t:
            pass
        captured2 = capsys.readouterr()
        assert "[reuse]" in captured2.out

    def test_duration_reflects_elapsed_time(self):
        """Duration should reflect at least the time spent inside the block."""
        t = Timer("sleeper")
        with t:
            time.sleep(0.05)
        assert t.duration.total_seconds() >= 0.04

    def test_start_and_end_attributes(self):
        """Timer should set start and end attributes after use."""
        t = Timer("attrs")
        with t:
            pass
        assert hasattr(t, "start")
        assert hasattr(t, "end")
        assert t.end >= t.start

    def test_exit_does_not_suppress_exception(self):
        """Timer.__exit__ should return None/falsy so exceptions propagate."""
        t = Timer("nosuppress")
        t.__enter__()
        result = t.__exit__(ValueError, ValueError("test"), None)
        # __exit__ returns None (falsy) — exception is not suppressed
        assert not result

    def test_disabled_exit_does_not_suppress_exception(self):
        """Disabled Timer.__exit__ should also not suppress exceptions."""
        t = Timer("disabled_nosuppress", disable_timer=True)
        t.__enter__()
        result = t.__exit__(RuntimeError, RuntimeError("test"), None)
        assert not result

    def test_unicode_description(self, capsys):
        """Timer should handle Unicode characters in description."""
        with Timer("测试 タイマー"):
            pass
        captured = capsys.readouterr()
        assert "[测试 タイマー]" in captured.out

    def test_sequential_timers(self, capsys):
        """Multiple sequential timers should each produce output."""
        for name in ["first", "second", "third"]:
            with Timer(name):
                pass
        captured = capsys.readouterr()
        assert "[first]" in captured.out
        assert "[second]" in captured.out
        assert "[third]" in captured.out

    def test_memory_output_format(self, capsys):
        """Memory usage should be reported in GB."""
        with Timer("mem_check"):
            pass
        captured = capsys.readouterr()
        assert "GB" in captured.out

    def test_exception_type_in_failure_message(self, capsys):
        """Different exception types should be correctly reported."""
        with pytest.raises(TypeError):
            with Timer("type_err"):
                raise TypeError("bad type")
        captured = capsys.readouterr()
        assert "TypeError" in captured.out

        with pytest.raises(KeyError):
            with Timer("key_err"):
                raise KeyError("missing")
        captured2 = capsys.readouterr()
        assert "KeyError" in captured2.out
