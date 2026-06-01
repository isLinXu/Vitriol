"""
Telemetry and Metrics System for Vitriol.

Provides monitoring through:
- Prometheus metrics export
- Performance tracking
- Health checks
- Custom metrics
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Single metric value."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics.

    Supports:
    - Counters (monotonically increasing)
    - Gauges (arbitrary values)
    - Histograms (value distributions)
    - Summaries (percentiles)
    """

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, MetricValue] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def counter(self, name: str, value: float = 1, labels: Optional[Dict] = None):
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to add
            labels: Metric labels
        """
        with self._lock:
            label_key = self._labels_to_key(labels)
            self._counters[f"{name}{label_key}"] += value

    def gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """
        Set a gauge metric.

        Args:
            name: Metric name
            value: Current value
            labels: Metric labels
        """
        with self._lock:
            label_key = self._labels_to_key(labels)
            self._gauges[f"{name}{label_key}"] = MetricValue(
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )

    def histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """
        Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        with self._lock:
            label_key = self._labels_to_key(labels)
            self._histograms[f"{name}{label_key}"].append(value)

            # Limit history
            if len(self._histograms[f"{name}{label_key}"]) > 10000:
                self._histograms[f"{name}{label_key}"] = self._histograms[f"{name}{label_key}"][-5000:]

    def ingest_dict(
        self,
        prefix: str,
        data: Dict[str, Any],
        labels: Optional[Dict] = None,
        *,
        numeric_kind: str = "gauge",
        sep: str = "_",
    ) -> None:
        """
        Ingest a nested dict into metrics by flattening numeric leaves.

        This is a minimal Phase2 helper intended for exporting runtime stats
        (e.g. cache_hooks / turboquant / kv_store) into MetricsCollector.

        Behavior:
        - Only dict nesting is traversed; non-dict containers are ignored.
        - Numeric leaves (int/float, excluding bool) are exported.
        - By default values are exported as gauges. If numeric_kind="counter",
          values are ingested as counter deltas (increment).

        Args:
            prefix: Metric name prefix, e.g. "kv".
            data: Nested dict of stats.
            labels: Optional labels applied to all exported metrics.
            numeric_kind: "gauge" (default) or "counter".
            sep: Path separator for flattened metric names.
        """
        if not isinstance(data, dict):
            return

        kind = str(numeric_kind or "gauge").lower()
        for path, value in _iter_numeric_leaves(data):
            suffix = sep.join(_sanitize_metric_token(p) for p in path if p)
            name = _sanitize_metric_token(prefix) if prefix else ""
            if suffix:
                name = f"{name}{sep}{suffix}" if name else suffix
            if not name:
                continue

            v = float(value)
            if kind == "counter":
                self.counter(name, v, labels=labels)
            else:
                self.gauge(name, v, labels=labels)

    def _labels_to_key(self, labels: Optional[Dict]) -> str:
        """Convert labels to string key."""
        if not labels:
            return ""
        return "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}"

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics in Prometheus format."""
        with self._lock:
            metrics = {}

            # Counters
            for name, value in self._counters.items():
                metrics[name] = {
                    "type": "counter",
                    "value": value
                }

            # Gauges
            for name, metric in self._gauges.items():
                metrics[name] = {
                    "type": "gauge",
                    "value": metric.value,
                    "timestamp": metric.timestamp
                }

            # Histograms
            for name, values in self._histograms.items():
                if values:
                    metrics[name] = {
                        "type": "histogram",
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "p50": sorted(values)[len(values) // 2],
                        "p95": sorted(values)[int(len(values) * 0.95)],
                        "p99": sorted(values)[int(len(values) * 0.99)]
                    }

            return metrics

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for name, data in self.get_metrics().items():
            metric_type = data.get("type", "unknown")

            if metric_type == "counter":
                lines.append(f"# TYPE {name.split('{')[0]} counter")
                lines.append(f"{name} {data['value']}")

            elif metric_type == "gauge":
                lines.append(f"# TYPE {name.split('{')[0]} gauge")
                lines.append(f"{name} {data['value']}")

            elif metric_type == "histogram":
                base_name = name.split('{')[0]
                lines.append(f"# TYPE {base_name} histogram")
                lines.append(f"{name}_count {data['count']}")
                lines.append(f"{name}_sum {data['sum']}")

        return "\n".join(lines)


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start_time
        self.collector.histogram(f"{self.name}_duration_seconds", self.duration, self.labels)


class HealthChecker:
    """
    Health check system.

    Supports multiple health check types:
    - Liveness: Is the service running?
    - Readiness: Is the service ready to accept requests?
    - Dependencies: Are external dependencies healthy?
    """

    def __init__(self):
        self.checks: Dict[str, Callable[[], tuple]] = {}
        self._cache: Dict[str, tuple] = {}
        self._cache_time: Dict[str, float] = {}
        self._cache_ttl = 5.0  # seconds

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], tuple],
        cache_ttl: Optional[float] = None
    ):
        """
        Register a health check.

        Args:
            name: Check name
            check_fn: Function returning (status, details)
            cache_ttl: Cache time-to-live
        """
        self.checks[name] = check_fn
        if cache_ttl:
            self._cache_ttl = cache_ttl

    def check(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run health check(s).

        Args:
            name: Specific check to run, or None for all

        Returns:
            Health check results
        """
        results = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }

        checks_to_run = {name: self.checks[name]} if name else self.checks

        for check_name, check_fn in checks_to_run.items():
            # Check cache
            if check_name in self._cache:
                cache_age = time.time() - self._cache_time[check_name]
                if cache_age < self._cache_ttl:
                    status, details = self._cache[check_name]
                    results["checks"][check_name] = {
                        "status": status,
                        "details": details,
                        "cached": True
                    }
                    continue

            # Run check
            try:
                status, details = check_fn()

                # Update cache
                self._cache[check_name] = (status, details)
                self._cache_time[check_name] = time.time()

                results["checks"][check_name] = {
                    "status": status,
                    "details": details,
                    "cached": False
                }

                if status != "healthy":
                    results["status"] = "degraded"

            except Exception as e:
                results["checks"][check_name] = {
                    "status": "unhealthy",
                    "details": str(e),
                    "cached": False
                }
                results["status"] = "unhealthy"

        return results


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_health_checker: Optional[HealthChecker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def _is_number(value: Any) -> bool:
    # bool is a subclass of int; treat it as non-numeric for metrics export.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _iter_numeric_leaves(data: Dict[str, Any]) -> Iterable[tuple[List[str], float]]:
    """
    Yield (path, value) for numeric leaves in a nested dict.

    Note: for Phase2 we only traverse dicts. Lists/tuples/etc are ignored.
    """

    def walk(node: Any, path: List[str]) -> Iterable[tuple[List[str], float]]:
        if isinstance(node, dict):
            for k, v in node.items():
                yield from walk(v, [*path, str(k)])
            return
        if _is_number(node):
            yield (path, float(node))

    yield from walk(data, [])


def _sanitize_metric_token(token: str) -> str:
    """
    Sanitize a metric name token to be Prometheus-friendly-ish.

    We keep ASCII letters/digits/underscore; everything else becomes underscore.
    """
    out = []
    for ch in str(token):
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9") or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")
