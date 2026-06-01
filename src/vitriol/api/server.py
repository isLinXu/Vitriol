"""
REST API Server for Vitriol.

Provides HTTP endpoints for:
- Model generation
- Architecture search
- Job status tracking
- Batch generation

⚠️ EXPERIMENTAL:
This API module is optional and not part of the default installation.
Install via: `pip install "vitriol[api]"`
"""

import asyncio
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..config.manager import build_generation_config
from ..config.settings import get_config
from ..logging.logger import get_logger
from ..version import __version__

logger = get_logger("vitriol.api")
_APP_STARTED_AT = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Log Streaming Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class LogStreamQueue(logging.Handler):
    """
    A log handler that collects logs into a bounded deque for streaming.
    This allows the /stream/logs endpoint to broadcast real-time logs.
    """
    def __init__(self, maxlen: int = 1000):
        super().__init__()
        self._queue: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        """Add a log record to the stream queue."""
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            if record.exc_info:
                formatter = self.formatter or logging.Formatter()
                log_entry["exception"] = formatter.formatException(record.exc_info)
            with self._lock:
                self._queue.append(log_entry)
        except Exception:
            # Silently drop logs on error to avoid disrupting the main application
            pass

    def get_logs(self, since_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all log entries, optionally filtered by timestamp."""
        with self._lock:
            logs = list(self._queue)
        if since_timestamp:
            try:
                cutoff = datetime.fromisoformat(since_timestamp.replace('Z', '+00:00'))
                logs = [
                    item
                    for item in logs
                    if datetime.fromisoformat(item["timestamp"].replace('Z', '+00:00')) > cutoff
                ]
            except (ValueError, TypeError):
                pass  # Return all logs if timestamp parsing fails
        return logs


# Global log stream handler
_log_stream_handler = LogStreamQueue(maxlen=1000)

# Add the handler to the root logger so it captures all logs
logging.getLogger().addHandler(_log_stream_handler)


def _result_to_dict(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "output_dir": getattr(result, "output_dir", None),
        "manifest_path": getattr(result, "manifest_path", None),
        "index_path": getattr(result, "index_path", None),
        "total_size": getattr(result, "total_size", None),
        "generated_at": getattr(result, "generated_at", None),
    }


def _model_dump(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

# API Models

class GenerateRequest(BaseModel):
    """Request model for weight generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_id": "gpt2",
                "strategy": "ultra",
                "dtype": "bfloat16",
            }
        }
    )

    model_id: str = Field(..., description="HuggingFace model ID")
    strategy: str = Field("compact", description="Generation strategy")
    dtype: str = Field("bfloat16", description="Data type")
    max_shard_size: str = Field("5GB", description="Max shard size")
    output_dir: Optional[str] = Field(None, description="Output directory")
    trust_remote_code: Optional[bool] = Field(
        None,
        description="Override trust_remote_code for this request; uses server defaults when omitted",
    )
    allow_network: Optional[bool] = Field(
        None,
        description="Whether HuggingFace downloads/network access is allowed; uses server defaults when omitted",
    )
    local_files_only: Optional[bool] = Field(
        None,
        description="Force local_files_only=True for HF loading (implied when allow_network=False)",
    )


class GenerateResponse(BaseModel):
    """Response model for generation."""
    job_id: str
    status: str
    message: str
    output_dir: Optional[str] = None
    estimated_time: Optional[int] = None


class NASRequest(BaseModel):
    """Request model for architecture search."""
    algorithm: str = Field("evolutionary", description="Search algorithm")
    n_iterations: int = Field(50, ge=10, le=1000)
    population_size: int = Field(20, ge=5, le=100)
    dataset: Optional[str] = Field(None, description="Evaluation dataset")
    allow_network: Optional[bool] = Field(
        None,
        description="Whether network access is allowed during NAS (e.g., tokenizer/model downloads); uses server defaults when omitted",
    )
    local_files_only: Optional[bool] = Field(
        None,
        description="Force local_files_only=True during NAS (implied when allow_network=False)",
    )


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    architecture: str
    parameters: int
    size_mb: float
    supported: bool


class SystemStatus(BaseModel):
    """System status information."""
    status: str
    version: str
    uptime: float
    active_jobs: int
    queue_size: int
    system_info: Dict[str, Any]


# Job tracking
active_jobs: Dict[str, Dict[str, Any]] = {}
job_queue: asyncio.Queue = asyncio.Queue()

# Create FastAPI app
app = FastAPI(
    title="Vitriol API",
    description="REST API for Vitriol model generation, NAS search, job tracking, and batch generation.",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware — default restricts to localhost; set VITRIOL_API_CORS_ORIGINS=* for open access
_cors_origins = os.environ.get("VITRIOL_API_CORS_ORIGINS", "http://localhost,http://127.0.0.1")
allow_origins = [o.strip() for o in _cors_origins.split(",")] if _cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Rate limiting middleware
class RateLimiter:
    """Simple in-memory sliding-window rate limiter.

    Configure via environment variable ``VITRIOL_API_RATE_LIMIT``.
    Format: ``<count>/<window>`` where window is ``second``, ``minute``, or ``hour``.
    Example: ``60/minute`` (default), ``100/hour``, ``10/second``.
    """

    def __init__(self):
        self._storage: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        self._max_requests: int = 60
        self._window_seconds: int = 60

        raw = os.environ.get("VITRIOL_API_RATE_LIMIT", "60/minute")
        self._parse_limit(raw)

    def _parse_limit(self, raw: str) -> None:
        try:
            count_str, window = raw.strip().split("/")
            self._max_requests = int(count_str)
            window = window.lower()
            if window.startswith("second"):
                self._window_seconds = 1
            elif window.startswith("minute"):
                self._window_seconds = 60
            elif window.startswith("hour"):
                self._window_seconds = 3600
            else:
                logger.warning("Unknown rate limit window '%s', using 'minute'", window)
                self._window_seconds = 60
        except (ValueError, AttributeError) as exc:
            logger.warning("Invalid VITRIOL_API_RATE_LIMIT value '%s': %s", raw, exc)
            self._max_requests = 60
            self._window_seconds = 60

    def _get_client_ip(self, request: Request) -> str:
        trust_proxy_headers = os.environ.get("VITRIOL_API_TRUST_PROXY_HEADERS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if trust_proxy_headers:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip.strip()
        if request.client:
            return request.client.host
        return "unknown"

    def check(self, request: Request) -> None:
        """Raise HTTP 429 if the client has exceeded the rate limit."""
        client_ip = self._get_client_ip(request)
        now = time.time()
        window_start = now - self._window_seconds

        with self._lock:
            dq = self._storage[client_ip]
            # Remove timestamps outside the window
            while dq and dq[0] < window_start:
                dq.popleft()
            if len(dq) >= self._max_requests:
                retry_after = int(dq[0] - window_start) if dq else self._window_seconds
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {self._max_requests} requests per {self._window_seconds}s.",
                    headers={"Retry-After": str(retry_after)},
                )
            dq.append(now)
            # Cleanup: remove old entries periodically (every 100 requests)
            if len(dq) % 100 == 0:
                while dq and dq[0] < window_start:
                    dq.popleft()


_rate_limiter = RateLimiter()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    try:
        _rate_limiter.check(request)
    except HTTPException:
        raise  # Re-raise 429 exceptions
    except Exception as exc:
        logger.warning("Rate limiter error: %s", exc)
    response = await call_next(request)
    return response


# Authentication
def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _matches_api_key(candidate: Optional[str], valid_keys: List[str]) -> bool:
    if not candidate:
        return False
    return any(hmac.compare_digest(candidate, str(valid_key)) for valid_key in valid_keys)


def verify_api_key(
    api_key: Optional[str] = Query(None, description="Deprecated: API key query parameter"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
):
    """Verify API key from X-API-Key, Bearer token, or legacy query parameter."""
    config = get_config()
    if config.get("security.api_key_required"):
        valid_keys = config.get("security.api_keys", []) or []
        candidate = x_api_key or _bearer_token(authorization) or api_key
        if not _matches_api_key(candidate, valid_keys):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return candidate
    return x_api_key or _bearer_token(authorization) or api_key


# Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "Vitriol API",
        "version": __version__,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status", response_model=SystemStatus)
async def get_status(_api_key: Optional[str] = Depends(verify_api_key)):
    """Get system status."""
    import psutil

    uptime_seconds = max(0.0, time.monotonic() - _APP_STARTED_AT)

    return SystemStatus(
        status="running",
        version=__version__,
        uptime=uptime_seconds,
        active_jobs=len(active_jobs),
        queue_size=job_queue.qsize(),
        system_info={
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
            "system_boot_time": psutil.boot_time(),
        }
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_weights(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    _api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Generate model weights.

    This endpoint starts a weight generation job asynchronously.
    Use the returned job_id to check status.
    """
    job_id = str(uuid.uuid4())

    # Create job
    job = {
        "id": job_id,
        "type": "generation",
        "status": "queued",
        "request": _model_dump(request),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0
    }

    active_jobs[job_id] = job
    await job_queue.put(job)

    # Start background processing
    background_tasks.add_task(process_generation_job, job_id)

    logger.info(f"Generation job queued: {job_id}")

    return GenerateResponse(
        job_id=job_id,
        status="queued",
        message="Job queued successfully",
        estimated_time=300  # 5 minutes estimate
    )


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Get job status."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return active_jobs[job_id]


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """List all jobs."""
    jobs = list(active_jobs.values())

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    jobs = sorted(jobs, key=lambda x: x["created_at"], reverse=True)

    return {"jobs": jobs[:limit], "total": len(active_jobs)}


@app.post("/nas/search")
async def start_nas_search(
    request: NASRequest,
    background_tasks: BackgroundTasks,
    _api_key: Optional[str] = Depends(verify_api_key)
):
    """Start architecture search."""
    job_id = str(uuid.uuid4())

    job = {
        "id": job_id,
        "type": "nas",
        "status": "queued",
        "request": _model_dump(request),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0
    }

    active_jobs[job_id] = job
    await job_queue.put(job)

    background_tasks.add_task(process_nas_job, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "NAS search started"
    }


@app.get("/models")
async def list_supported_models(_api_key: Optional[str] = Depends(verify_api_key)):
    """List supported model families, adapters, and known models.

    Dynamically generated from Vitriol's internal data sources:
    - ``DEFAULT_FAMILIES`` (evolution tree)
    - ``FALLBACK_PARAMS`` (known architecture parameters)
    - ``AdapterRegistry`` (registered model adapters)
    - ``STRATEGY_REGISTRY`` (available weight generation strategies)
    """
    from ..adapters.registry import AdapterRegistry
    from ..evolution.tree_builder import DEFAULT_FAMILIES, FALLBACK_PARAMS
    from ..utils.strategy_discovery import discover_strategy_names

    # ── Families ──────────────────────────────────────────────────────
    families = []
    for fam_name, fam_data in DEFAULT_FAMILIES.items():
        members = list(fam_data.get("members", {}).keys())
        # Collect innovations across all members
        innovations_set: set = set()
        for _model_id, inn_list in fam_data.get("innovations", {}).items():
            for inn in inn_list:
                innovations_set.add(inn.name if hasattr(inn, "name") else str(inn))
        families.append({
            "name": fam_name,
            "root": fam_data.get("root", ""),
            "members_count": len(members),
            "members": members,
            "key_innovations": sorted(innovations_set),
        })

    # ── Adapters ──────────────────────────────────────────────────────
    adapters = AdapterRegistry.discover_builtin_adapter_metadata()

    # ── Known models (from FALLBACK_PARAMS) ───────────────────────────
    known_models = []
    for model_id, params in FALLBACK_PARAMS.items():
        hidden = params.get("hidden_size", 0)
        n_layers = params.get("num_hidden_layers", 0)
        model_type = params.get("model_type", "unknown")
        # Rough parameter estimate: hidden^2 * n_layers * 12 (transformer rule of thumb)
        rough_params = int(hidden * hidden * n_layers * 12) if hidden > 0 else 0
        known_models.append(ModelInfo(
            model_id=model_id,
            architecture=model_type,
            parameters=rough_params,
            size_mb=rough_params * 2 / (1024 ** 2),  # bfloat16 estimate
            supported=True,
        ))

    # ── Strategies summary ────────────────────────────────────────────
    strategy_names = discover_strategy_names()

    return {
        "models": known_models,
        "families": families,
        "adapters": adapters,
        "strategies": strategy_names,
        "notes": {
            "trust_remote_code": "Default False; can be enabled per request or via CLI --trust-remote-code",
            "source": "Dynamically generated from Vitriol internal registries",
            "families_count": len(families),
            "known_models_count": len(known_models),
            "adapters_count": len(adapters),
        },
    }


@app.get("/models/families")
async def list_model_families(_api_key: Optional[str] = Depends(verify_api_key)):
    """List model architecture families from the evolution tree."""
    from ..evolution.tree_builder import DEFAULT_FAMILIES

    families = []
    for fam_name, fam_data in DEFAULT_FAMILIES.items():
        members = list(fam_data.get("members", {}).keys())
        innovations_set: set = set()
        for _model_id, inn_list in fam_data.get("innovations", {}).items():
            for inn in inn_list:
                innovations_set.add(inn.name if hasattr(inn, "name") else str(inn))
        families.append({
            "name": fam_name,
            "root": fam_data.get("root", ""),
            "members_count": len(members),
            "members": members,
            "key_innovations": sorted(innovations_set),
        })

    return {"families": families, "total": len(families)}


@app.get("/models/adapters")
async def list_model_adapters(_api_key: Optional[str] = Depends(verify_api_key)):
    """List registered model adapters and their capabilities."""
    from ..adapters.registry import AdapterRegistry
    adapters = AdapterRegistry.discover_builtin_adapter_metadata()

    return {"adapters": adapters, "total": len(adapters)}


@app.get("/strategies")
async def list_strategies(_api_key: Optional[str] = Depends(verify_api_key)):
    """List available generation strategies."""
    from ..strategies import STRATEGY_REGISTRY

    strategies = []
    for name, strategy_class in STRATEGY_REGISTRY.items():
        instance = strategy_class()
        caps = instance.capabilities
        strategies.append({
            "name": name,
            "description": caps.description,
            "supports_safetensors": caps.supports_safetensors,
            "supports_training": caps.supports_training,
            "max_compression": caps.max_compression_ratio
        })

    return {"strategies": strategies}


@app.get("/stream/logs")
async def stream_logs(
    since: Optional[str] = Query(None, description="ISO timestamp to get logs since"),
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Stream logs in real-time using the LogStreamQueue handler.

    This endpoint streams all log messages captured by the LogStreamQueue handler,
    which is attached to the root logger to capture all logs from the application.
    """
    async def log_generator():
        # Send initial batch of recent logs
        recent_logs = _log_stream_handler.get_logs(since_timestamp=since)
        for log_entry in recent_logs:
            yield f"data: {json.dumps(log_entry)}\n\n"

        # Then stream new logs as they come in
        last_idx = len(recent_logs)
        while True:
            await asyncio.sleep(0.5)  # Poll every 500ms
            all_logs = _log_stream_handler.get_logs(since_timestamp=since)
            if len(all_logs) > last_idx:
                for log_entry in all_logs[last_idx:]:
                    yield f"data: {json.dumps(log_entry)}\n\n"
                last_idx = len(all_logs)

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch Generation Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class BatchGenerateRequest(BaseModel):
    """Request model for batch generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "models": [
                    {"model_id": "gpt2", "output_dir": "./output/gpt2"},
                    {"model_id": "llama-7b", "output_dir": "./output/llama-7b"},
                ],
                "default_strategy": "compact",
                "parallel": False,
            }
        }
    )

    models: List[Dict[str, Any]] = Field(
        ...,
        description="List of models to generate"
    )
    default_strategy: str = Field("compact", description="Default generation strategy")
    default_dtype: str = Field("bfloat16", description="Default data type")
    parallel: bool = Field(False, description="Reserved for future parallel batch execution; currently must be false")
    trust_remote_code: Optional[bool] = Field(
        None,
        description="Override trust_remote_code for the batch; uses server defaults when omitted",
    )


class BatchJobStatus(BaseModel):
    """Status of a batch job."""
    batch_id: str
    total_models: int
    completed: int
    failed: int
    status: str  # pending, running, completed, failed
    results: List[Dict[str, Any]]


@app.post("/batch/generate", response_model=BatchJobStatus)
async def start_batch_generation(
    request: BatchGenerateRequest,
    background_tasks: BackgroundTasks,
    _api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Start a batch generation job.

    Generate minimal weights for multiple models in a single request.
    """
    if request.parallel:
        raise HTTPException(status_code=400, detail="parallel batch generation is not supported yet")

    batch_id = str(uuid.uuid4())

    job = {
        "id": batch_id,
        "type": "batch",
        "status": "queued",
        "total_models": len(request.models),
        "completed": 0,
        "failed": 0,
        "results": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": _model_dump(request)
    }

    active_jobs[batch_id] = job

    background_tasks.add_task(process_batch_job, batch_id, request)

    return BatchJobStatus(
        batch_id=batch_id,
        total_models=len(request.models),
        completed=0,
        failed=0,
        status="queued",
        results=[]
    )


@app.get("/batch/{batch_id}", response_model=BatchJobStatus)
async def get_batch_status(batch_id: str, _api_key: Optional[str] = Depends(verify_api_key)):
    """Get the status of a batch generation job."""
    if batch_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")

    job = active_jobs[batch_id]
    return BatchJobStatus(
        batch_id=batch_id,
        total_models=job.get("total_models", 0),
        completed=job.get("completed", 0),
        failed=job.get("failed", 0),
        status=job.get("status", "unknown"),
        results=job.get("results", [])
    )


async def process_batch_job(batch_id: str, request: BatchGenerateRequest):
    """Process a batch generation job."""
    job = active_jobs.get(batch_id)
    if not job:
        return

    job["status"] = "running"
    logger.info(f"Starting batch generation job: {batch_id}")

    try:
        from ..core.generator import MinimalWeightGenerator

        total = len(request.models)

        for i, model_spec in enumerate(request.models):
            try:
                model_id = model_spec.get("model_id")
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ValueError("model_id is required for each batch model")

                output_dir = model_spec.get("output_dir") or f"./output/{model_id.split('/')[-1]}"
                strategy = model_spec.get("strategy", request.default_strategy)
                dtype = model_spec.get("dtype", request.default_dtype)
                trust_remote_code = model_spec.get("trust_remote_code", request.trust_remote_code)

                logger.info(f"Batch [{i + 1}/{total}]: Generating {model_id}")

                overrides = {
                    "strategy": strategy,
                    "dtype": dtype,
                    "max_shard_size": model_spec.get("max_shard_size", "5GB"),
                }
                if trust_remote_code is not None:
                    overrides["trust_remote_code"] = bool(trust_remote_code)

                config = build_generation_config(overrides=overrides)

                generator = MinimalWeightGenerator(
                    model_id=model_id,
                    output_dir=output_dir,
                    config=config
                )

                result = generator.generate()
                result_data = _result_to_dict(result)
                total_size = result_data.get("total_size") or 0
                result_output_dir = result_data.get("output_dir") or output_dir

                job["results"].append({
                    "model_id": model_id,
                    "status": "success",
                    "output_dir": result_output_dir,
                    "total_size": total_size,
                    "size_mb": total_size / (1024 ** 2),
                    "result": result_data,
                })
                job["completed"] += 1

            except Exception as e:
                model_id = model_spec.get("model_id")
                logger.error(f"Batch [{i + 1}/{total}]: Failed {model_id}: {e}")
                job["results"].append({
                    "model_id": model_id,
                    "status": "failed",
                    "error": str(e)
                })
                job["failed"] += 1

            # Update progress
            job["progress"] = int((i + 1) / total * 100)

        job["status"] = "completed" if job["failed"] == 0 else "completed_with_errors"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Batch job completed: {batch_id}, "
            f"success={job['completed']}, failed={job['failed']}"
        )

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Batch job failed: {batch_id}", exc_info=e)


# Background job processing

async def process_generation_job(job_id: str):
    """Process a generation job."""
    job = active_jobs.get(job_id)
    if not job:
        return

    job["status"] = "running"
    logger.info(f"Starting generation job: {job_id}")

    try:
        # Import here to avoid circular dependencies
        from ..core.generator import MinimalWeightGenerator

        request = job["request"]

        overrides = {
            "strategy": request.get("strategy", "compact"),
            "dtype": request.get("dtype", "bfloat16"),
            "max_shard_size": request.get("max_shard_size", "5GB"),
        }
        # Fail closed: remote model code must be enabled explicitly per request.
        overrides["trust_remote_code"] = bool(request.get("trust_remote_code", False))
        overrides["allow_network"] = bool(request.get("allow_network", True))
        overrides["local_files_only"] = bool(
            request.get("local_files_only", False) or (not overrides["allow_network"])
        )

        config = build_generation_config(overrides=overrides)

        generator = MinimalWeightGenerator(
            model_id=request["model_id"],
            output_dir=request.get("output_dir", f"./output/{request['model_id'].split('/')[-1]}"),
            config=config,
        )

        result = generator.generate()
        result_data = _result_to_dict(result)
        result_data["security_context"] = getattr(config, "security_context", {})

        job["status"] = "completed"
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        job["output_dir"] = result_data.get("output_dir") or request.get("output_dir")
        job["result"] = result_data

        logger.info(f"Generation job completed: {job_id}")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Generation job failed: {job_id}", exc_info=e)


async def process_nas_job(job_id: str):
    """Process a NAS job."""
    job = active_jobs.get(job_id)
    if not job:
        return

    job["status"] = "running"
    logger.info(f"Starting NAS job: {job_id}")

    try:
        request = job.get("request") or {}
        n_iterations = int(request.get("n_iterations", 50))
        n_iterations = max(1, n_iterations)

        # NOTE: targeted_nas ConstraintOptimizer currently uses random search + constraint filtering.
        # This endpoint provides an observable loop: streaming progress per iteration and exporting
        # the best gene/metrics as artifacts.
        from ..nas.search_space import LLMSearchSpace
        from ..nas.targeted_nas import ConstraintOptimizer, ObjectiveType, OptimizationTarget

        optimizer = ConstraintOptimizer(
            constraints=[],
            objectives=[
                OptimizationTarget(objective_type=ObjectiveType.MINIMIZE_PARAMS, weight=1.0),
            ],
        )
        search_space = LLMSearchSpace()

        best_gene = None
        best_score = -float("inf")
        best_metrics = {}

        # base_evaluator is not used by ConstraintOptimizer.optimize yet; keep it for future extensibility.
        for i in range(n_iterations):
            gene = search_space.sample()
            ok, _violated = optimizer.check_constraints(gene)
            if not ok:
                continue

            score = optimizer.evaluate_objectives(gene, base_score=0.0)
            if score > best_score:
                best_score = score
                best_gene = gene
                from ..nas.targeted_nas import MetricsCalculator

                best_metrics = MetricsCalculator.calculate_all(gene)

            job["progress"] = int((i + 1) / n_iterations * 100)
            # yield control (so the server can respond to /jobs polling promptly)
            await asyncio.sleep(0)

        if best_gene is None:
            raise RuntimeError("NAS search did not find any feasible architecture (constraints too strict?)")

        result = {
            "best_gene": best_gene.to_dict(),
            "best_score": float(best_score),
            "best_metrics": best_metrics,
            "n_iterations": n_iterations,
        }

        artifacts_dir = job.get("artifacts_dir")
        if artifacts_dir:
            try:
                os.makedirs(artifacts_dir, exist_ok=True)
                out_path = os.path.join(artifacts_dir, f"nas-result-{job_id}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                result["artifacts_path"] = out_path
            except Exception as e:
                logger.warning(f"Failed to write NAS artifacts: {e}")

        job["status"] = "completed"
        job["result"] = result

        logger.info(f"NAS job completed: {job_id}")

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"NAS job failed: {job_id}", exc_info=e)


# CLI entry point

def main():
    """Run API server."""
    config = get_config()
    host = os.environ.get("VITRIOL_API_HOST", "127.0.0.1")
    port = int(os.environ.get("VITRIOL_API_PORT", "8000"))

    uvicorn.run(
        "vitriol.api.server:app",
        host=host,
        port=port,
        reload=config.is_development(),
        log_level=config.get("system.log_level", "info").lower()
    )


if __name__ == "__main__":
    main()
