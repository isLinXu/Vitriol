"""Tests for vitriol.logging.logger module."""
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vitriol.logging.logger import (
    StructuredLogFormatter,
    ColoredConsoleFormatter,
    AsyncLogHandler,
    PerformanceTracker,
    VitriolLogger,
    get_logger,
    init_logger,
)


# ─────────────────────────────────────────────────────────────
# StructuredLogFormatter
# ─────────────────────────────────────────────────────────────

class TestStructuredLogFormatter:
    def test_format_basic(self):
        fmt = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        out = fmt.format(record)
        data = json.loads(out)
        assert data["level"] == "INFO"
        assert data["message"] == "hello"
        assert "timestamp" in data

    def test_format_with_extra_data(self):
        fmt = StructuredLogFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        record.extra_data = {"request_id": "123"}
        out = fmt.format(record)
        data = json.loads(out)
        assert data["request_id"] == "123"

    def test_format_with_exception(self):
        fmt = StructuredLogFormatter()
        try:
            raise ValueError("oops")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="fail", args=(), exc_info=exc_info,
            )
        out = fmt.format(record)
        data = json.loads(out)
        assert "exception" in data
        assert "oops" in data["exception"]


# ─────────────────────────────────────────────────────────────
# ColoredConsoleFormatter
# ─────────────────────────────────────────────────────────────

class TestColoredConsoleFormatter:
    def test_format_includes_color(self):
        fmt = ColoredConsoleFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="bad", args=(), exc_info=None,
        )
        out = fmt.format(record)
        assert "\033[" in out  # ANSI escape code
        assert "ERROR" in out
        assert "bad" in out

    def test_format_debug_color(self):
        fmt = ColoredConsoleFormatter("%(levelname)s - %(message)s")
        record = logging.LogRecord(
            name="test", level=logging.DEBUG, pathname="", lineno=0,
            msg="dbg", args=(), exc_info=None,
        )
        out = fmt.format(record)
        assert "\033[36m" in out  # Cyan for DEBUG


# ─────────────────────────────────────────────────────────────
# AsyncLogHandler
# ─────────────────────────────────────────────────────────────

class TestAsyncLogHandler:
    def test_emit_queues_record(self):
        target = MagicMock()
        handler = AsyncLogHandler(target, max_queue_size=100)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        handler.emit(record)
        # Give the background thread a moment to process
        import time
        time.sleep(0.1)
        target.emit.assert_called()
        handler.close()

    def test_emit_drops_when_full(self):
        target = MagicMock()
        handler = AsyncLogHandler(target, max_queue_size=1)
        # Fill the queue and force a drop
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        # Put something in the queue so the next emit may drop
        handler.queue.put_nowait(record)
        # This should not raise
        handler.emit(record)
        handler.close()


# ─────────────────────────────────────────────────────────────
# PerformanceTracker
# ─────────────────────────────────────────────────────────────

class TestPerformanceTracker:
    def test_track_decorator_success(self):
        logger = MagicMock()
        pt = PerformanceTracker(logger)

        @pt.track("my_op")
        def do_work():
            return 42

        result = do_work()
        assert result == 42
        assert "my_op" in pt.metrics
        assert len(pt.metrics["my_op"]) == 1

    def test_track_decorator_failure(self):
        logger = MagicMock()
        pt = PerformanceTracker(logger)

        @pt.track("my_op")
        def do_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            do_fail()

        assert "my_op" in pt.metrics
        assert len(pt.metrics["my_op"]) == 1

    def test_get_stats(self):
        logger = MagicMock()
        pt = PerformanceTracker(logger)
        pt.metrics["op"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        stats = pt.get_stats("op")
        assert stats["count"] == 5
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5
        assert stats["mean"] == 0.3

    def test_get_stats_missing(self):
        logger = MagicMock()
        pt = PerformanceTracker(logger)
        assert pt.get_stats("missing") == {}

    def test_slow_operation_warning(self):
        logger = MagicMock()
        pt = PerformanceTracker(logger)

        @pt.track("slow_op")
        def do_slow():
            import time
            time.sleep(1.1)
            return 1

        do_slow()
        logger.warning.assert_called()


# ─────────────────────────────────────────────────────────────
# VitriolLogger
# ─────────────────────────────────────────────────────────────

class TestVitriolLogger:
    def test_init(self):
        vl = VitriolLogger("test_logger")
        assert vl.name == "test_logger"
        assert len(vl.logger.handlers) >= 1

    def test_log_levels(self):
        vl = VitriolLogger("test_levels")
        # Should not raise
        vl.debug("debug msg", extra_key="v")
        vl.info("info msg")
        vl.warning("warn msg")
        vl.error("err msg")
        vl.critical("crit msg")

    def test_add_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vl = VitriolLogger("test_file")
            vl.add_file_handler(str(Path(tmpdir) / "app.log"))
            assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in vl.logger.handlers)

    def test_add_file_handler_unstructured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vl = VitriolLogger("test_file2")
            vl.add_file_handler(str(Path(tmpdir) / "app.log"), structured=False)
            # Should have a plain formatter
            fh = [h for h in vl.logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)][0]
            assert isinstance(fh.formatter, logging.Formatter)

    def test_set_and_clear_context(self):
        vl = VitriolLogger("test_ctx")
        vl.set_context(request_id="r1", operation="op1", user_id="u1")
        # Just verify it doesn't raise
        vl.clear_context()

    def test_track_performance(self):
        vl = VitriolLogger("test_perf")

        @vl.track_performance("timed_op")
        def work():
            return 42

        work()
        stats = vl.get_performance_stats()
        assert "timed_op" in stats

    def test_get_performance_stats_empty(self):
        vl = VitriolLogger("test_perf_empty")
        assert vl.get_performance_stats() == {}


# ─────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────

class TestModuleHelpers:
    def test_get_logger_singleton(self):
        # get_logger caches the instance
        l1 = get_logger("vitriol_test")
        l2 = get_logger("vitriol_test")
        assert l1 is l2

    def test_init_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "test.log")
            vl = init_logger("vitriol_init", log_file=log_path, structured=True)
            assert isinstance(vl, VitriolLogger)
            assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in vl.logger.handlers)
