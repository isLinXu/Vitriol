"""
Interactive Visualization Dashboard for Vitriol.

This module provides a real-time web-based dashboard for monitoring
and visualizing model generation, architecture search, and analysis.

Features:
- Real-time progress tracking
- Interactive architecture visualization
- Performance metrics charts
- Live logs and notifications
"""

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime

# Web framework
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DashboardEvent:
    """Event for dashboard display."""
    timestamp: float
    event_type: str  # 'progress', 'metric', 'log', 'completion'
    data: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "data": self.data
        }


@dataclass
class GenerationMetrics:
    """Metrics for weight generation."""
    total_params: int = 0
    generated_params: int = 0
    current_shard: int = 0
    total_shards: int = 0
    generation_speed: float = 0.0  # params/sec
    eta_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    compression_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class NASMetrics:
    """Metrics for architecture search."""
    iteration: int = 0
    total_iterations: int = 0
    best_score: float = 0.0
    current_score: float = 0.0
    architectures_evaluated: int = 0
    search_space_size: int = 0
    estimated_time_remaining: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class DashboardDataStore:
    """
    Thread-safe data store for dashboard.

    Maintains all metrics and events for display.
    """

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self.events: List[DashboardEvent] = []
        self.lock = threading.Lock()

        # Current state
        self.generation_metrics = GenerationMetrics()
        self.nas_metrics = NASMetrics()
        self.active_operation: Optional[str] = None
        # Phase2 Task6: minimal dashboard run_id support
        self.current_run_id: Optional[str] = None
        self.logs: List[str] = []

        # Callbacks for real-time updates
        self.subscribers: List[callable] = []

    def add_event(self, event: DashboardEvent) -> None:
        """Add event to store."""
        with self.lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events.pop(0)

        # Notify subscribers
        for callback in self.subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.debug("Dashboard event subscriber callback failed: %s", e)

    def update_generation_metrics(self, metrics: GenerationMetrics) -> None:
        """Update generation metrics."""
        with self.lock:
            self.generation_metrics = metrics

        self.add_event(DashboardEvent(
            timestamp=time.time(),
            event_type="metric",
            data={"type": "generation", "metrics": metrics.to_dict()}
        ))

    def update_nas_metrics(self, metrics: NASMetrics) -> None:
        """Update NAS metrics."""
        with self.lock:
            self.nas_metrics = metrics

        self.add_event(DashboardEvent(
            timestamp=time.time(),
            event_type="metric",
            data={"type": "nas", "metrics": metrics.to_dict()}
        ))

    def add_log(self, message: str, level: str = "info") -> None:
        """Add log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"

        with self.lock:
            self.logs.append(log_entry)
            if len(self.logs) > 100:
                self.logs.pop(0)

        self.add_event(DashboardEvent(
            timestamp=time.time(),
            event_type="log",
            data={"message": log_entry, "level": level}
        ))

    def set_active_operation(self, operation: Optional[str]) -> None:
        """Set currently active operation."""
        with self.lock:
            self.active_operation = operation

        if operation:
            self.add_event(DashboardEvent(
                timestamp=time.time(),
                event_type="operation",
                data={"operation": operation, "status": "started"}
            ))

    def set_run_id(self, run_id: Optional[str]) -> None:
        """Set the current run id (thread-safe).

        Backward compatible: does not affect existing state keys, only adds a new one.
        Emits a 'run' event so SSE clients can react to run changes.
        """
        with self.lock:
            self.current_run_id = run_id

        self.add_event(
            DashboardEvent(
                timestamp=time.time(),
                event_type="run",
                data={"run_id": run_id},
            )
        )

    def get_state(self) -> Dict[str, Any]:
        """Get current state for dashboard."""
        with self.lock:
            return {
                "generation": self.generation_metrics.to_dict(),
                "nas": self.nas_metrics.to_dict(),
                "active_operation": self.active_operation,
                "run_id": self.current_run_id,
                "logs": self.logs[-20:],  # Last 20 logs
                "events": [e.to_dict() for e in self.events[-50:]]  # Last 50 events
            }

    def subscribe(self, callback: callable) -> None:
        """Subscribe to real-time updates."""
        self.subscribers.append(callback)

    def unsubscribe(self, callback: callable) -> None:
        """Unsubscribe from updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard."""

    data_store: Optional[DashboardDataStore] = None

    def log_message(self, format, *args) -> None:
        # Suppress default logging
        pass

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == '/':
            self._serve_dashboard()
        elif self.path == '/api/state':
            self._serve_api_state()
        elif self.path == '/api/events':
            self._serve_events_stream()
        elif self.path.startswith('/static/'):
            self._serve_static()
        else:
            self.send_error(404)

    def _serve_dashboard(self):
        """Serve main dashboard HTML."""
        html = self._get_dashboard_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_api_state(self):
        """Serve current state as JSON."""
        if self.data_store:
            state = self.data_store.get_state()
        else:
            state = {}

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(state).encode())

    def _serve_events_stream(self):
        """Serve server-sent events stream."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if not self.data_store:
            return

        # Queue for this connection
        event_queue = queue.Queue()

        def on_event(event: DashboardEvent) -> None:
            event_queue.put(event)

        self.data_store.subscribe(on_event)

        try:
            while True:
                try:
                    event = event_queue.get(timeout=1)
                    data = json.dumps(event.to_dict())
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Send keepalive
                    self.wfile.write(b":\n\n")
                    self.wfile.flush()
        except Exception as e:
            logger.debug("SSE connection closed: %s", e)
        finally:
            self.data_store.unsubscribe(on_event)

    def _serve_static(self):
        """Serve static files."""
        self.send_error(404)  # Not implemented for simplicity

    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML content."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vitriol Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }

        .header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid #334155;
        }

        .header h1 {
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }

        .card h2 {
            font-size: 1.1rem;
            color: #94a3b8;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #334155;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            color: #94a3b8;
            font-size: 0.9rem;
        }

        .metric-value {
            font-family: 'SF Mono', Monaco, monospace;
            font-weight: 600;
            color: #60a5fa;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-active {
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }

        .status-idle {
            background: rgba(148, 163, 184, 0.2);
            color: #94a3b8;
        }

        .logs {
            background: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.85rem;
            max-height: 300px;
            overflow-y: auto;
        }

        .log-entry {
            padding: 0.25rem 0;
            color: #cbd5e1;
        }

        .log-entry.error {
            color: #ef4444;
        }

        .log-entry.warning {
            color: #f59e0b;
        }

        .log-entry.info {
            color: #60a5fa;
        }

        .chart-container {
            height: 200px;
            background: #0f172a;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .pulse {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏛️ Vitriol Dashboard</h1>
    </div>

    <div class="container">
        <div class="grid">
            <div class="card">
                <h2>Status</h2>
                <div class="metric">
                    <span class="metric-label">Active Operation</span>
                    <span id="active-operation" class="status-badge status-idle">Idle</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span id="uptime" class="metric-value">00:00:00</span>
                </div>
            </div>

            <div class="card">
                <h2>Generation Progress</h2>
                <div class="metric">
                    <span class="metric-label">Parameters</span>
                    <span id="gen-params" class="metric-value">0 / 0</span>
                </div>
                <div class="progress-bar">
                    <div id="gen-progress" class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Shards</span>
                    <span id="gen-shards" class="metric-value">0 / 0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Speed</span>
                    <span id="gen-speed" class="metric-value">0 params/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ETA</span>
                    <span id="gen-eta" class="metric-value">--:--</span>
                </div>
            </div>

            <div class="card">
                <h2>NAS Progress</h2>
                <div class="metric">
                    <span class="metric-label">Iteration</span>
                    <span id="nas-iter" class="metric-value">0 / 0</span>
                </div>
                <div class="progress-bar">
                    <div id="nas-progress" class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Best Score</span>
                    <span id="nas-best" class="metric-value">0.0000</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Score</span>
                    <span id="nas-current" class="metric-value">0.0000</span>
                </div>
            </div>

            <div class="card">
                <h2>System</h2>
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span id="sys-memory" class="metric-value">0 MB</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Compression</span>
                    <span id="sys-compression" class="metric-value">0%</span>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Live Logs</h2>
            <div id="logs" class="logs">
                <div class="log-entry info">Dashboard initialized. Waiting for events...</div>
            </div>
        </div>
    </div>

    <script>
        const startTime = Date.now();

        // Update uptime
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600).toString().padStart(2, '0');
            const minutes = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
            const seconds = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('uptime').textContent = `${hours}:${minutes}:${seconds}`;
        }, 1000);

        // Fetch state periodically
        async function fetchState() {
            try {
                const response = await fetch('/api/state');
                const state = await response.json();
                updateDashboard(state);
            } catch (err) {
                console.error('Failed to fetch state:', err);
            }
        }

        function updateDashboard(state) {
            // Update generation metrics
            if (state.generation) {
                const gen = state.generation;
                document.getElementById('gen-params').textContent =
                    `${formatNumber(gen.generated_params)} / ${formatNumber(gen.total_params)}`;
                document.getElementById('gen-shards').textContent =
                    `${gen.current_shard} / ${gen.total_shards}`;
                document.getElementById('gen-speed').textContent =
                    `${formatNumber(gen.generation_speed)} params/s`;
                document.getElementById('gen-eta').textContent = formatTime(gen.eta_seconds);

                const progress = gen.total_params > 0 ?
                    (gen.generated_params / gen.total_params * 100) : 0;
                document.getElementById('gen-progress').style.width = `${progress}%`;

                document.getElementById('sys-memory').textContent = `${gen.memory_usage_mb.toFixed(1)} MB`;
                document.getElementById('sys-compression').textContent = `${(gen.compression_ratio * 100).toFixed(1)}%`;
            }

            // Update NAS metrics
            if (state.nas) {
                const nas = state.nas;
                document.getElementById('nas-iter').textContent =
                    `${nas.iteration} / ${nas.total_iterations}`;
                document.getElementById('nas-best').textContent = nas.best_score.toFixed(4);
                document.getElementById('nas-current').textContent = nas.current_score.toFixed(4);

                const progress = nas.total_iterations > 0 ?
                    (nas.iteration / nas.total_iterations * 100) : 0;
                document.getElementById('nas-progress').style.width = `${progress}%`;
            }

            // Update operation status
            if (state.active_operation) {
                const opEl = document.getElementById('active-operation');
                opEl.textContent = state.active_operation;
                opEl.className = 'status-badge status-active pulse';
            } else {
                const opEl = document.getElementById('active-operation');
                opEl.textContent = 'Idle';
                opEl.className = 'status-badge status-idle';
            }

            // Update logs — use createElement + textContent to avoid XSS
            if (state.logs && state.logs.length > 0) {
                const logsEl = document.getElementById('logs');
                // Clear safely
                while (logsEl.firstChild) logsEl.removeChild(logsEl.firstChild);
                for (const log of state.logs) {
                    const level = log.includes('[ERROR]') ? 'error' :
                                 log.includes('[WARNING]') ? 'warning' : 'info';
                    const entry = document.createElement('div');
                    entry.className = `log-entry ${level}`;
                    entry.textContent = log;
                    logsEl.appendChild(entry);
                }
                logsEl.scrollTop = logsEl.scrollHeight;
            }
        }

        function formatNumber(num) {
            if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(2) + 'K';
            return num.toString();
        }

        function formatTime(seconds) {
            if (!seconds || seconds === 0) return '--:--';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            if (mins >= 60) {
                const hours = Math.floor(mins / 60);
                return `${hours}h ${mins % 60}m`;
            }
            return `${mins}m ${secs.toString().padStart(2, '0')}s`;
        }

        // Poll for updates
        setInterval(fetchState, 1000);
        fetchState(); // Initial fetch

        // Connect to event stream
        const evtSource = new EventSource('/api/events');
        evtSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Event:', data);
        };
    </script>
</body>
</html>'''


class DashboardServer:
    """
    Dashboard web server.

    Provides HTTP endpoints for dashboard data and SSE for real-time updates.
    """

    def __init__(self, data_store: DashboardDataStore, port: int = 8080):
        """
        Initialize dashboard server.

        Args:
            data_store: DashboardDataStore instance
            port: HTTP port
        """
        self.data_store = data_store
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

        # Set data store on handler class
        DashboardRequestHandler.data_store = data_store

    def start(self, blocking: bool = False) -> None:
        """Start the dashboard server."""
        if self.server:
            logger.warning("Server already running")
            return

        try:
            self.server = HTTPServer(('localhost', self.port), DashboardRequestHandler)

            if blocking:
                logger.info(f"Dashboard server running at http://localhost:{self.port}")
                self.server.serve_forever()
            else:
                self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                self.thread.start()
                logger.info(f"Dashboard server started at http://localhost:{self.port}")
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"Port {self.port} in use, trying {self.port + 1}")
                self.port += 1
                self.start(blocking)
            else:
                raise

    def stop(self) -> None:
        """Stop the dashboard server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            logger.info("Dashboard server stopped")


class VitriolDashboard:
    """
    Main dashboard interface for Vitriol.

    Provides easy integration with Vitriol workflows.
    """

    def __init__(self, port: int = 8080):
        """
        Initialize dashboard.

        Args:
            port: HTTP port for dashboard
        """
        self.data_store = DashboardDataStore()
        self.server = DashboardServer(self.data_store, port)
        self._started = False

    def start(self) -> None:
        """Start the dashboard."""
        if not self._started:
            self.server.start(blocking=False)
            self._started = True
            self.data_store.add_log("Dashboard started", "info")

    def stop(self) -> None:
        """Stop the dashboard."""
        self.server.stop()
        self._started = False

    def update_generation(self, metrics: GenerationMetrics) -> None:
        """Update generation metrics."""
        self.data_store.update_generation_metrics(metrics)

    def update_nas(self, metrics: NASMetrics) -> None:
        """Update NAS metrics."""
        self.data_store.update_nas_metrics(metrics)

    def log(self, message: str, level: str = "info") -> None:
        """Add log message."""
        self.data_store.add_log(message, level)

    def set_operation(self, operation: Optional[str]) -> None:
        """Set active operation."""
        self.data_store.set_active_operation(operation)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
