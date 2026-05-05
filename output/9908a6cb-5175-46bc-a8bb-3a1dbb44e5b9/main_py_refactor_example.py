"""
大朝议 III · main.py 拆分重构示例
演示如何将 2105 行巨石拆分为 APIRouter 模块

目录结构（目标）：
backend/
  main.py                  # < 300 行：FastAPI app 组装
  api/
    __init__.py           # 导出所有 router
    health.py             # 健康检查
    decree.py             # 圣旨相关
    rooms.py              # 房间管理
    history.py            # 历史记录
    statistics.py         # 统计数据
    notifications.py      # 通知系统
    events.py             # 随机事件
    tasks.py              # 奏章系统
    knowledge.py          # 知识库
    reports.py            # 报告系统
"""

# ============================================================================
# 示例 1: backend/api/health.py
# ============================================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import time

router = APIRouter(prefix="", tags=["健康检查"])


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, str]


@router.get("/health")
async def health_check():
    """简单健康检查"""
    return {"status": "healthy", "timestamp": time.time()}


@router.get("/health/ready")
async def readiness_check():
    """就绪检查（检查依赖服务）"""
    services = {
        "database": "ok",  # 实际应检查 Redis/SQLite 连接
        "llm": "ok",       # 实际应检查 LLM API
    }
    
    if all(v == "ok" for v in services.values()):
        return HealthResponse(status="ready", timestamp=time.time(), services=services)
    else:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "services": services})


@router.get("/health/verbose")
async def verbose_health_check():
    """详细健康检查（包含版本、配置等）"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "3.0.0-MVP",
        "services": {
            "redis": "ok",
            "celery": "ok",
            "vector_store": "ok",
        },
        "config": {
            "max_concurrent": 10,
            "rate_limit": "60/minute",
        }
    }


# ============================================================================
# 示例 2: backend/api/decree.py
# ============================================================================

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api", tags=["圣旨系统"])


class DecreeRequest(BaseModel):
    content: str
    room_id: Optional[str] = None
    user_id: str = "emperor"


class DecreeResponse(BaseModel):
    task_id: str
    status: str
    message: str


@router.post("/decree")
async def issue_decree(decree: DecreeRequest, background_tasks: BackgroundTasks):
    """发布圣旨"""
    from backend.edict_graph import edict_app
    from backend.scheduler import TaskQueue
    import uuid
    
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    # 将任务加入队列（实际应用中应通过 Celery）
    # background_tasks.add_task(edict_app.invoke, {"decree": decree.content})
    
    return DecreeResponse(
        task_id=task_id,
        status="pending",
        message=f"圣旨已下达，任务ID: {task_id}"
    )


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    # 实际应从 Redis/数据库查询
    return {
        "task_id": task_id,
        "status": "processing",
        "progress": 50,
        "current_node": "zhongshu",
    }


# ============================================================================
# 示例 3: backend/api/__init__.py
# ============================================================================

"""
API 路由聚合模块
"""
from . import health, decree, rooms, history, statistics
from . import notifications, events, tasks, knowledge, reports

__all__ = [
    "health",
    "decree",
    "rooms",
    "history",
    "statistics",
    "notifications",
    "events",
    "tasks",
    "knowledge",
    "reports",
]


# ============================================================================
# 示例 4: backend/main.py（重构后的精简版）
# ============================================================================

"""
FastAPI 应用入口（重构版）
- 应用组装与中间件配置
- WebSocket 路由（因其特殊性，保留在 main.py）
- 生命周期管理
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from contextlib import asynccontextmanager
import asyncio

# 导入所有 API 路由
from backend.api import (
    health, decree, rooms, history, statistics,
    notifications, events, tasks, knowledge, reports
)
from backend.config import REDIS_URL, RATE_LIMIT_PER_MINUTE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("🚀 大朝议启动...")
    # 可在此初始化 Redis 连接池、Celery 等
    
    yield
    
    # 关闭时清理
    print("🛑 大朝议关闭...")


# 创建 FastAPI 应用
app = FastAPI(
    title="大朝议 III",
    description="古风 Multi-Agent 协作演示平台",
    version="3.0.0-MVP",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置限流
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# 注册所有 API 路由
app.include_router(health.router)
app.include_router(decree.router)
app.include_router(rooms.router)
app.include_router(history.router)
app.include_router(statistics.router)
app.include_router(notifications.router)
app.include_router(events.router)
app.include_router(tasks.router)
app.include_router(knowledge.router)
app.include_router(reports.router)


# WebSocket 路由（因其特殊性，保留在 main.py）
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 连接入口"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            # 处理消息...
            await websocket.send_json({"type": "ack", "message": "received"})
    
    except WebSocketDisconnect:
        print("WebSocket disconnected")


# 根路由
@app.get("/")
async def root():
    return {
        "project": "大朝议 III",
        "version": "3.0.0-MVP",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================================================
# 迁移步骤（供参考）
# ============================================================================

"""
1. 创建 backend/api/ 目录：
   mkdir -p backend/api

2. 从 main.py 中提取各域路由到对应文件：
   - 健康检查相关 → api/health.py
   - /api/decree, /api/task/{id} → api/decree.py
   - /api/rooms/* → api/rooms.py
   - /api/history/* → api/history.py
   - /api/statistics/* → api/statistics.py
   - /api/notifications/* → api/notifications.py
   - /api/events/* → api/events.py
   - /api/tasks/* → api/tasks.py
   - /api/knowledge/* → api/knowledge.py
   - /api/reports/* → api/reports.py

3. 每个文件创建 APIRouter：
   router = APIRouter(prefix="/api", tags=["XXX系统"])

4. 将依赖（如 TaskQueue、event_manager）改为显式导入，不依赖全局变量

5. 在 main.py 中 include_router()

6. 运行测试确保无破坏：
   pytest tests/backend/test_main.py -v

7. 手动冒烟测试关键端点：
   curl http://localhost:8000/health
   curl http://localhost:8000/api/decree -X POST -H "Content-Type: application/json" -d '{"content":"test"}'
"""
