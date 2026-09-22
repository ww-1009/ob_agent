"""FastAPI 应用装配。create_app 便于测试注入 stub 模型/客户端。"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

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
from app.tools.sql import get_meta_db_executor

def create_app(
    settings: Settings | None = None,
    *,
    model=None,
    ocp=None,
    meta_db_executor=None,
    memory: MemoryRuntime | None = None,
) -> FastAPI:
    settings = settings if settings is not None else load_settings()
    # fail closed：开了访问控制却没配令牌，宁可起不来也不要"以为开了其实没开"
    if settings.auth.enabled and not settings.auth.token.strip():
        raise RuntimeError(
            "auth.enabled=true 但 auth.token 为空：请配置 AUTH_TOKEN 或 auth.token（或关闭 auth.enabled）"
        )
    ocp = ocp if ocp is not None else get_ocp_client(settings)
    meta_db_executor = meta_db_executor if meta_db_executor is not None else get_meta_db_executor(settings)
    # LLM 未注入时：仅当配置齐全才构建；否则留 None，/api/chat 会返回 503 清晰提示。
    if model is None and settings.llm.is_configured:
        model = build_chat_model(settings.llm)
    tools = build_tools(ocp, settings.sql_ro, send_row_data=settings.agent.send_row_data)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 注入的运行时（测试/静态模式）优先；否则按配置打开，失败降级为无记忆。
        runtime = memory
        if runtime is None and settings.memory.enabled:
            runtime = await open_memory(settings.memory)
        app.state.memory = runtime
        try:
            yield
        finally:
            if runtime is not None and memory is None:
                await runtime.aclose()

    app = FastAPI(title="OceanBase DB Agent Backend", lifespan=lifespan)
    app.state.settings = settings
    app.state.model = model
    app.state.tools = tools
    # 供路由在 lifespan 之外（如直接调用）也能安全读取
    app.state.memory = memory
    # /api/health 免令牌（探活），其余业务路由统一挂令牌依赖
    guarded = [Depends(require_token)]
    app.include_router(health_router)
    app.include_router(chat_router, dependencies=guarded)
    app.include_router(make_confirm_router(), dependencies=guarded)
    app.include_router(make_threads_router(), dependencies=guarded)
    app.include_router(make_audit_router(), dependencies=guarded)
    return app


app = create_app()
