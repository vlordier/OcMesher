"""Tests for ocmesher/utils/timer.py — Timer context manager."""

import io
import sys
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
