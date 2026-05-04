"""Tests for vitriol.telemetry.metrics module."""
import time

import pytest

from vitriol.telemetry.metrics import (
    MetricValue,
    MetricsCollector,
    PerformanceTimer,
    HealthChecker,
    get_metrics_collector,
    get_health_checker,
)


# ─────────────────────────────────────────────────────────────
# MetricValue
# ─────────────────────────────────────────────────────────────

class TestMetricValue:
    def test_dataclass(self):
        mv = MetricValue(value=42.0, timestamp=time.time(), labels={"env": "test"})
        assert mv.value == 42.0
        assert mv.labels == {"env": "test"}


# ─────────────────────────────────────────────────────────────
# MetricsCollector
# ─────────────────────────────────────────────────────────────

class TestMetricsCollector:
    def test_counter(self):
        mc = MetricsCollector()
        mc.counter("requests", value=1)
        mc.counter("requests", value=2)
        metrics = mc.get_metrics()
        assert metrics["requests"]["value"] == 3

    def test_counter_with_labels(self):
        mc = MetricsCollector()
        mc.counter("requests", value=1, labels={"method": "GET"})
        mc.counter("requests", value=1, labels={"method": "POST"})
        metrics = mc.get_metrics()
        assert len(metrics) == 2

    def test_gauge(self):
        mc = MetricsCollector()
        mc.gauge("temperature", 36.5)
        metrics = mc.get_metrics()
        assert metrics["temperature"]["value"] == 36.5
        assert metrics["temperature"]["type"] == "gauge"

    def test_histogram(self):
        mc = MetricsCollector()
        mc.histogram("latency", 0.1)
        mc.histogram("latency", 0.2)
        mc.histogram("latency", 0.3)
        metrics = mc.get_metrics()
        assert metrics["latency"]["type"] == "histogram"
        assert metrics["latency"]["count"] == 3
        assert metrics["latency"]["avg"] == pytest.approx(0.2)

    def test_histogram_rollup(self):
        mc = MetricsCollector()
        for i in range(10001):
            mc.histogram("big_hist", float(i))
        # After exceeding 10000, should roll up to last 5000
        metrics = mc.get_metrics()
        assert metrics["big_hist"]["count"] == 5000

    def test_labels_to_key_empty(self):
        mc = MetricsCollector()
        assert mc._labels_to_key(None) == ""
        assert mc._labels_to_key({}) == ""

    def test_labels_to_key_sorted(self):
        mc = MetricsCollector()
        key = mc._labels_to_key({"b": "2", "a": "1"})
        assert key == '{a="1",b="2"}'

    def test_to_prometheus_format(self):
        mc = MetricsCollector()
        mc.counter("requests_total", 100)
        mc.gauge("active_connections", 5)
        mc.histogram("request_duration_seconds", 0.1)
        text = mc.to_prometheus_format()
        assert "requests_total" in text
        assert "active_connections" in text
        assert "request_duration_seconds_count" in text
        assert "TYPE" in text


# ─────────────────────────────────────────────────────────────
# PerformanceTimer
# ─────────────────────────────────────────────────────────────

class TestPerformanceTimer:
    def test_context_manager(self):
        mc = MetricsCollector()
        with PerformanceTimer(mc, "my_op"):
            time.sleep(0.01)
        metrics = mc.get_metrics()
        assert "my_op_duration_seconds" in metrics
        assert metrics["my_op_duration_seconds"]["count"] == 1
        assert metrics["my_op_duration_seconds"]["avg"] > 0

    def test_context_manager_with_labels(self):
        mc = MetricsCollector()
        with PerformanceTimer(mc, "my_op", labels={"region": "us"}):
            pass
        metrics = mc.get_metrics()
        assert "my_op_duration_seconds{region=\"us\"}" in metrics


# ─────────────────────────────────────────────────────────────
# HealthChecker
# ─────────────────────────────────────────────────────────────

class TestHealthChecker:
    def test_register_check(self):
        hc = HealthChecker()
        hc.register_check("db", lambda: ("healthy", "ok"))
        assert "db" in hc.checks

    def test_check_all_healthy(self):
        hc = HealthChecker()
        hc.register_check("db", lambda: ("healthy", "connected"))
        hc.register_check("cache", lambda: ("healthy", "warm"))
        result = hc.check()
        assert result["status"] == "healthy"
        assert result["checks"]["db"]["status"] == "healthy"
        assert result["checks"]["cache"]["status"] == "healthy"

    def test_check_one_unhealthy(self):
        hc = HealthChecker()
        hc.register_check("db", lambda: ("healthy", "ok"))
        hc.register_check("cache", lambda: ("unhealthy", "cold"))
        result = hc.check()
        assert result["status"] == "degraded"
        assert result["checks"]["cache"]["status"] == "unhealthy"

    def test_check_specific(self):
        hc = HealthChecker()
        hc.register_check("db", lambda: ("healthy", "ok"))
        hc.register_check("cache", lambda: ("unhealthy", "cold"))
        result = hc.check("db")
        assert result["status"] == "healthy"
        assert len(result["checks"]) == 1

    def test_check_exception(self):
        hc = HealthChecker()
        hc.register_check("db", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result = hc.check()
        assert result["status"] == "unhealthy"
        assert "boom" in result["checks"]["db"]["details"]

    def test_check_cache_hit(self):
        hc = HealthChecker()
        call_count = 0

        def check_fn():
            nonlocal call_count
            call_count += 1
            return ("healthy", f"call {call_count}")

        hc.register_check("db", check_fn, cache_ttl=10.0)
        r1 = hc.check("db")
        r2 = hc.check("db")
        assert call_count == 1  # cached
        assert r2["checks"]["db"]["cached"] is True

    def test_check_cache_expired(self):
        hc = HealthChecker()
        call_count = 0

        def check_fn():
            nonlocal call_count
            call_count += 1
            return ("healthy", "ok")

        hc.register_check("db", check_fn, cache_ttl=0.001)
        r1 = hc.check("db")
        time.sleep(0.02)
        r2 = hc.check("db")
        assert call_count == 2  # cache expired


# ─────────────────────────────────────────────────────────────
# Global Instances
# ─────────────────────────────────────────────────────────────

class TestGlobalInstances:
    def test_get_metrics_collector_singleton(self):
        mc1 = get_metrics_collector()
        mc2 = get_metrics_collector()
        assert mc1 is mc2

    def test_get_health_checker_singleton(self):
        hc1 = get_health_checker()
        hc2 = get_health_checker()
        assert hc1 is hc2
