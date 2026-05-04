"""Tests for viz/dashboard module."""

import threading
import time


from vitriol.viz.dashboard import (
    DashboardDataStore,
    DashboardEvent,
    DashboardServer,
    GenerationMetrics,
    NASMetrics,
    VitriolDashboard,
)


# ─────────────────────────────────────────────────────────────────────────────
# DashboardEvent Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardEvent:
    """Tests for DashboardEvent dataclass."""

    def test_to_dict(self):
        event = DashboardEvent(
            timestamp=123.0,
            event_type="test",
            data={"key": "value"},
        )
        d = event.to_dict()
        assert d["timestamp"] == 123.0
        assert d["event_type"] == "test"
        assert d["data"] == {"key": "value"}


# ─────────────────────────────────────────────────────────────────────────────
# GenerationMetrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerationMetrics:
    """Tests for GenerationMetrics dataclass."""

    def test_to_dict(self):
        metrics = GenerationMetrics(
            total_params=1000,
            generated_params=500,
            current_shard=2,
            total_shards=10,
        )
        d = metrics.to_dict()
        assert d["total_params"] == 1000
        assert d["generated_params"] == 500
        assert d["generation_speed"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# NASMetrics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNASMetrics:
    """Tests for NASMetrics dataclass."""

    def test_to_dict(self):
        metrics = NASMetrics(
            iteration=5,
            total_iterations=100,
            best_score=0.95,
        )
        d = metrics.to_dict()
        assert d["iteration"] == 5
        assert d["best_score"] == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# DashboardDataStore Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardDataStore:
    """Tests for DashboardDataStore."""

    def test_add_event(self):
        store = DashboardDataStore()
        event = DashboardEvent(timestamp=1.0, event_type="test", data={})
        store.add_event(event)
        assert len(store.events) == 1

    def test_event_limit(self):
        store = DashboardDataStore(max_events=5)
        for i in range(10):
            store.add_event(DashboardEvent(timestamp=float(i), event_type="test", data={}))
        assert len(store.events) == 5
        assert store.events[0].timestamp == 5.0  # Oldest events removed

    def test_update_generation_metrics(self):
        store = DashboardDataStore()
        metrics = GenerationMetrics(total_params=1000, generated_params=500)
        store.update_generation_metrics(metrics)
        assert store.generation_metrics.total_params == 1000
        assert len(store.events) == 1
        assert store.events[0].event_type == "metric"

    def test_update_nas_metrics(self):
        store = DashboardDataStore()
        metrics = NASMetrics(iteration=10, best_score=0.9)
        store.update_nas_metrics(metrics)
        assert store.nas_metrics.iteration == 10
        assert len(store.events) == 1

    def test_add_log(self):
        store = DashboardDataStore()
        store.add_log("Test message", "info")
        assert len(store.logs) == 1
        assert "Test message" in store.logs[0]
        assert "INFO" in store.logs[0]

    def test_log_limit(self):
        store = DashboardDataStore()
        for i in range(150):
            store.add_log(f"Message {i}")
        assert len(store.logs) == 100

    def test_set_active_operation(self):
        store = DashboardDataStore()
        store.set_active_operation("generation")
        assert store.active_operation == "generation"
        assert len(store.events) == 1
        assert store.events[0].event_type == "operation"

    def test_set_active_operation_none(self):
        store = DashboardDataStore()
        store.set_active_operation("generation")
        store.set_active_operation(None)
        assert store.active_operation is None

    def test_get_state(self):
        store = DashboardDataStore()
        store.update_generation_metrics(GenerationMetrics(total_params=100))
        store.add_log("Test")
        state = store.get_state()
        assert "generation" in state
        assert "nas" in state
        assert "logs" in state
        assert "events" in state
        assert state["active_operation"] is None

    def test_subscribe_and_unsubscribe(self):
        store = DashboardDataStore()
        received = []

        def callback(event):
            received.append(event)

        store.subscribe(callback)
        store.add_event(DashboardEvent(timestamp=1.0, event_type="test", data={}))
        assert len(received) == 1

        store.unsubscribe(callback)
        store.add_event(DashboardEvent(timestamp=2.0, event_type="test", data={}))
        assert len(received) == 1  # No more events

    def test_subscriber_exception_handled(self):
        store = DashboardDataStore()

        def bad_callback(event):
            raise RuntimeError("Oops")

        store.subscribe(bad_callback)
        # Should not raise
        store.add_event(DashboardEvent(timestamp=1.0, event_type="test", data={}))

    def test_thread_safety(self):
        store = DashboardDataStore()
        errors = []

        def worker():
            try:
                for i in range(50):
                    store.add_event(DashboardEvent(timestamp=time.time(), event_type="test", data={"i": i}))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store.events) == 200


# ─────────────────────────────────────────────────────────────────────────────
# DashboardServer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardServer:
    """Tests for DashboardServer."""

    def test_server_start_stop(self):
        store = DashboardDataStore()
        server = DashboardServer(store, port=0)  # port=0 for auto-assign
        # Don't actually start server to avoid port conflicts in tests
        assert server.data_store is store

    def test_port_increment_on_conflict(self):
        store = DashboardDataStore()
        server = DashboardServer(store, port=99999)
        assert server.port == 99999


# ─────────────────────────────────────────────────────────────────────────────
# VitriolDashboard Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVitriolDashboard:
    """Tests for VitriolDashboard."""

    def test_init(self):
        dashboard = VitriolDashboard(port=18080)
        assert dashboard._started is False
        assert dashboard.data_store is not None

    def test_context_manager(self):
        dashboard = VitriolDashboard(port=18081)
        with dashboard as d:
            assert d is dashboard
            assert dashboard._started is True
        assert dashboard._started is False

    def test_log(self):
        dashboard = VitriolDashboard(port=18082)
        dashboard.log("Test message", "warning")
        assert any("Test message" in log and "WARNING" in log for log in dashboard.data_store.logs)

    def test_set_operation(self):
        dashboard = VitriolDashboard(port=18083)
        dashboard.set_operation("testing")
        assert dashboard.data_store.active_operation == "testing"

    def test_update_generation(self):
        dashboard = VitriolDashboard(port=18084)
        metrics = GenerationMetrics(total_params=1000, generated_params=500)
        dashboard.update_generation(metrics)
        assert dashboard.data_store.generation_metrics.total_params == 1000

    def test_update_nas(self):
        dashboard = VitriolDashboard(port=18085)
        metrics = NASMetrics(iteration=5, best_score=0.95)
        dashboard.update_nas(metrics)
        assert dashboard.data_store.nas_metrics.best_score == 0.95
