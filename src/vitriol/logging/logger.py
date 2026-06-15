"""
Advanced Logging System for Vitriol.

Features:
- Structured logging (JSON)
- Multiple handlers (console, file, remote)
- Log rotation and compression
- Contextual logging
- Performance tracking
- Integration with monitoring
"""

import json
import logging
import logging.handlers
import queue
import sys
import threading
import time
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

# Context variables for contextual logging
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_operation: ContextVar[Optional[str]] = ContextVar('operation', default=None)
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context
        context = {
            "request_id": _request_id.get(),
            "operation": _operation.get(),
            "user_id": _user_id.get(),
        }
        if any(context.values()):
            log_data["context"] = {k: v for k, v in context.items() if v}

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""

    def __init__(self, target_handler: logging.Handler, max_queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.thread = threading.Thread(target=self._process_logs, daemon=True)
        self.thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Queue log record for async processing."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Drop log if queue is full
            pass

    def _process_logs(self):
        """Process logs in background thread."""
        while True:
            try:
                record = self.queue.get(timeout=1)
                self.target_handler.emit(record)
            except queue.Empty:
                continue
            except Exception:
                self.target_handler.handle(record)


class PerformanceTracker:
    """Track performance metrics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics: Dict[str, List[float]] = {}

    def track(self, operation: str) -> Any:
        """Decorator to track operation performance."""
        def decorator(func) -> Any:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    self._record(operation, duration, success=True)
                    return result
                except Exception as e:
                    duration = time.time() - start
                    self._record(operation, duration, success=False, error=str(e))
                    raise
            return wrapper
        return decorator

    def _record(self, operation: str, duration: float, success: bool, error: Optional[str] = None):
        """Record metric."""
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration)

        # Log slow operations
        if duration > 1.0:  # > 1 second
            self.logger.warning(
                f"Slow operation: {operation} took {duration:.2f}s",
                extra={'extra_data': {'duration': duration, 'success': success}}
            )

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}

        times = self.metrics[operation]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "p95": sorted(times)[int(len(times) * 0.95)],
            "p99": sorted(times)[int(len(times) * 0.99)],
        }


class VitriolLogger:
    """
    Advanced logger for Vitriol.

    Features:
        - Structured JSON logging
        - Multiple output targets
        - Async logging for performance
        - Context tracking
        - Performance monitoring
    """

    def __init__(self, name: str = "vitriol"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        self.performance = PerformanceTracker(self.logger)
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Setup default logging handlers."""
        # Console handler with colors
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(ColoredConsoleFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console)

    def add_file_handler(
        self,
        path: str,
        level: str = "DEBUG",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        structured: bool = True
    ) -> None:
        """
        Add rotating file handler.

        Args:
            path: Log file path
            level: Log level
            max_bytes: Max file size before rotation
            backup_count: Number of backups to keep
            structured: Use JSON formatting
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setLevel(getattr(logging, level.upper()))

        if structured:
            handler.setFormatter(StructuredLogFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        self.logger.addHandler(handler)

    def add_async_handler(self, target_handler: logging.Handler) -> None:
        """Add async wrapper to handler."""
        async_handler = AsyncLogHandler(target_handler)
        self.logger.addHandler(async_handler)

    def set_context(self, request_id: Optional[str] = None, operation: Optional[str] = None, user_id: Optional[str] = None) -> None:
        """Set logging context."""
        if request_id:
            _request_id.set(request_id)
        if operation:
            _operation.set(operation)
        if user_id:
            _user_id.set(user_id)

    def clear_context(self) -> None:
        """Clear logging context."""
        _request_id.set(None)
        _operation.set(None)
        _user_id.set(None)

    def debug(self, message: str, **extra) -> None:
        """Log debug message."""
        self.logger.debug(message, extra={'extra_data': extra} if extra else {})

    def info(self, message: str, **extra) -> None:
        """Log info message."""
        self.logger.info(message, extra={'extra_data': extra} if extra else {})

    def warning(self, message: str, **extra) -> None:
        """Log warning message."""
        self.logger.warning(message, extra={'extra_data': extra} if extra else {})

    def error(self, message: str, **extra) -> None:
        """Log error message."""
        self.logger.error(message, extra={'extra_data': extra} if extra else {})

    def critical(self, message: str, **extra) -> None:
        """Log critical message."""
        self.logger.critical(message, extra={'extra_data': extra} if extra else {})

    def track_performance(self, operation: str) -> Any:
        """Get performance tracker decorator."""
        return self.performance.track(operation)

    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get all performance statistics."""
        return {
            op: self.performance.get_stats(op)
            for op in self.performance.metrics.keys()
        }


# Global logger instance
_logger_instance: Optional[VitriolLogger] = None


def get_logger(name: str = "vitriol") -> VitriolLogger:
    """Get global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = VitriolLogger(name)
    return _logger_instance


def init_logger(
    name: str = "vitriol",
    log_file: Optional[str] = None,
    structured: bool = True
) -> VitriolLogger:
    """Initialize global logger."""
    global _logger_instance
    _logger_instance = VitriolLogger(name)

    if log_file:
        _logger_instance.add_file_handler(log_file, structured=structured)

    return _logger_instance
