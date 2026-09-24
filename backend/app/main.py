"""FastAPI 应用装配。create_app 便于测试注入 stub 模型/客户端。"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.agent.confirm import ConfirmationBroker
from app.agent.model import build_chat_model
from app.agent.tools import build_tools
from app.api.audit import make_audit_router
from app.api.chat import health_router
from app.api.chat import router as chat_router
from app.api.confirm import make_confirm_router
from app.api.deps import require_token
from app.api.threads import make_threads_router
from app.config import Settings, load_settings
from app.memory import MemoryRuntime, open_memory
from app.tools.ocp import get_ocp_client
from app.tools.sql.registry import close_all_executors

logger = logging.getLogger(__name__)


def _warn_if_multi_worker(settings: Settings) -> None:
    """HITL 确认通道与同一 thread 的串行锁都是进程内实现，多 worker 下会失效。

    worker 数在应用内无法可靠探测（uvicorn --workers 不会设置环境变量），因此这里只在
    常见的并发度变量存在且 >1 时给出明确告警，其余情况依赖 README 的单 worker 部署说明。
    """
    if not settings.agent.confirm_db_ops:
        return
    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        raw = (os.environ.get(name) or "").strip()
        if raw.isdigit() and int(raw) > 1:
            logger.warning(
                "检测到 %s=%s：人工确认（HITL）通道与同一 thread 的串行锁均为进程内实现，"
                "多 worker 下审批会返回 404/503、同一 thread 可能并发写检查点；请以 --workers 1 运行。",
                name,
                raw,
            )
            return


def create_app(
    settings: Settings | None = None,
    *,
    model=None,
    ocp=None,
    memory: MemoryRuntime | None = None,
) -> FastAPI:
    settings = settings if settings is not None else load_settings()
    # fail closed：开了访问控制却没配令牌，宁可起不来也不要"以为开了其实没开"
    if settings.auth.enabled and not settings.auth.token.strip():
        raise RuntimeError(
            "auth.enabled=true 但 auth.token 为空：请配置 AUTH_TOKEN 或 auth.token（或关闭 auth.enabled）"
        )
    ocp = ocp if ocp is not None else get_ocp_client(settings)
    _warn_if_multi_worker(settings)
    # LLM 未注入时：仅当配置齐全才构建；否则留 None，/api/chat 会返回 503 清晰提示。
    if model is None and settings.llm.is_configured:
        model = build_chat_model(settings.llm)
    tools = build_tools(ocp, settings.sql_ro, send_row_data=settings.agent.send_row_data)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 注入的运行时（测试/静态模式）优先；否则按配置打开（内部含重试），
        # 最终失败则降级为无记忆，并把原因挂到 app.state.memory_error 供 /api/health 展示。
        runtime = memory
        error: str | None = None
        if runtime is None and settings.memory.enabled:
            runtime, error = await open_memory(settings.memory)
        app.state.memory = runtime
        app.state.memory_error = error
        try:
            yield
        finally:
            if runtime is not None and memory is None:
                await runtime.aclose()
            # 真实 SQL 执行器持有长连接：进程退出前统一释放（无长连接时为空操作）
            close_all_executors()

    app = FastAPI(title="OceanBase DB Agent Backend", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = model
    app.state.tools = tools
    # 进程级共享确认 broker：request_id 全局唯一，跨会话隔离由 runner 的
    # ``fail_all(owner=producer_task)`` 保证（见 agent/runner.py）。每个 worker 一个实例，
    # 这正是「多 worker 下审批不可达」的边界所在（见 _warn_if_multi_worker）。
    app.state.confirm_broker = ConfirmationBroker()
    # 供路由在 lifespan 之外（如直接调用）也能安全读取
    app.state.memory = memory
    app.state.memory_error = None
    # /api/health 免令牌（探活），其余业务路由统一挂令牌依赖
    guarded = [Depends(require_token)]
    app.include_router(health_router)
    app.include_router(chat_router, dependencies=guarded)
    app.include_router(make_confirm_router(), dependencies=guarded)
    app.include_router(make_threads_router(), dependencies=guarded)
    app.include_router(make_audit_router(), dependencies=guarded)
    return app


app = create_app()
