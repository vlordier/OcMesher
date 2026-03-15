"""Tests for ``ocmesher.utils.timer``."""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import psutil
import pytest

from ocmesher.utils.timer import Timer

# ---------------------------------------------------------------------------
# Helpers - deterministic psutil stubs
# ---------------------------------------------------------------------------


class _FakeMemInfo:
    """Stub for psutil.Process.memory_info()."""

    rss = 512 * 1024 * 1024  # 0.50 GB


class _FakeProcess:
    """Stub for psutil.Process."""

    def __init__(self, _pid=None):
        pass

    def memory_info(self):
        return _FakeMemInfo()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTimerBasic:
    @patch("ocmesher.utils.timer.psutil.Process", _FakeProcess)
    def test_timer_prints_duration(self, caplog):
        with caplog.at_level(logging.INFO, logger="ocmesher.utils.timer"), Timer("test op"):
            time.sleep(0.01)
        assert "[test op] finished in" in caplog.text
        assert "GB" in caplog.text

    def test_timer_disabled(self, caplog):
        with caplog.at_level(logging.INFO, logger="ocmesher.utils.timer"), Timer("disabled", disable_timer=True):
            time.sleep(0.01)
        assert caplog.text == ""

    @patch("ocmesher.utils.timer.psutil.Process", _FakeProcess)
    def test_timer_records_duration(self):
        with Timer("duration test") as t:
            time.sleep(0.05)
        assert t.duration.total_seconds() >= 0.04

    @patch("ocmesher.utils.timer.psutil.Process", _FakeProcess)
    def test_timer_context_returns_self(self):
        with Timer("self test") as t:
            assert isinstance(t, Timer)

    def test_timer_disabled_returns_self(self):
        with Timer("disabled self", disable_timer=True) as t:
            assert isinstance(t, Timer)

    @patch("ocmesher.utils.timer.psutil.Process", _FakeProcess)
    def test_timer_output_contains_elapsed_time(self, caplog):
        with caplog.at_level(logging.INFO, logger="ocmesher.utils.timer"), Timer("elapsed test"):
            pass
        # Output must include a timedelta-style string, e.g. "0:00:00.000123"
        assert "0:00:0" in caplog.text


class TestTimerName:
    def test_name_formatted_with_brackets(self):
        t = Timer("my task")
        assert t.name == "[my task]"

    def test_disabled_has_no_name(self):
        t = Timer("my task", disable_timer=True)
        assert not hasattr(t, "name")


class TestTimerExceptionHandling:
    def test_prints_failure_on_exception(self, caplog):
        with (
            caplog.at_level(logging.WARNING, logger="ocmesher.utils.timer"),
            pytest.raises(ValueError, match="test error"),
            Timer("failing op"),
        ):
            raise ValueError("test error")  # noqa: EM101, TRY003
        assert "[failing op] failed with" in caplog.text
        assert "ValueError" in caplog.text

    def test_does_not_suppress_exception(self):
        with pytest.raises(RuntimeError), Timer("error test"):
            raise RuntimeError("boom")  # noqa: EM101

    def test_disabled_timer_does_not_affect_exception(self):
        with pytest.raises(ZeroDivisionError), Timer("disabled error", disable_timer=True):
            _ = 1 / 0


class TestTimerMemoryReporting:
    @patch("ocmesher.utils.timer.psutil.Process", _FakeProcess)
    def test_memory_output_format(self, caplog):
        with caplog.at_level(logging.INFO, logger="ocmesher.utils.timer"), Timer("mem test"):
            pass
        assert "0.50 GB" in caplog.text

    def test_handles_psutil_no_such_process(self, caplog):
        def _raise(_pid):
            raise psutil.NoSuchProcess(_pid)

        with (
            caplog.at_level(logging.INFO, logger="ocmesher.utils.timer"),
            patch("ocmesher.utils.timer.psutil.Process", side_effect=_raise),
            Timer("psutil error test"),
        ):
            pass
        assert "[psutil error test] finished in" in caplog.text
        assert "GB" not in caplog.text

    def test_handles_psutil_access_denied(self, caplog):
        def _raise(_pid):
            raise psutil.AccessDenied(_pid)

        with (
            caplog.at_level(logging.INFO, logger="ocmesher.utils.timer"),
            patch("ocmesher.utils.timer.psutil.Process", side_effect=_raise),
            Timer("access denied test"),
        ):
            pass
        assert "[access denied test] finished in" in caplog.text
        assert "GB" not in caplog.text


# ---------------------------------------------------------------------------
# Refactoring: __slots__ on Timer (commit 4)
# ---------------------------------------------------------------------------
class TestTimerSlots:
    def test_has_slots(self):
        """Timer must define __slots__ to reduce per-instance memory overhead."""
        assert hasattr(Timer, "__slots__")

    def test_no_instance_dict(self):
        """__slots__ prevents a per-instance __dict__ being created."""
        t = Timer("slot test")
        assert not hasattr(t, "__dict__")

    def test_slots_contain_expected_attrs(self):
        """All runtime attributes must be declared in __slots__."""
        expected = {"name", "start", "end", "duration", "disable_timer"}
        assert expected.issubset(set(Timer.__slots__))

    def test_cannot_set_arbitrary_attribute(self):
        """Setting an undeclared attribute must raise AttributeError."""
        t = Timer("slot guard")
        with pytest.raises(AttributeError):
            t.unexpected_attr = 42  # type: ignore[attr-defined]
