"""API 层共享依赖：访问控制令牌、会话记忆可用性、thread_id 校验。

集中在此避免各路由互相 import（threads/audit 都曾各自复制一份）。
"""
from __future__ import annotations

import re
import secrets

from fastapi import Header, HTTPException, Request

from app.memory import MemoryRuntime

# 会话 ID 白名单：同时用作 chat_message.thread_id 与检查点 thread_id
THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


async def require_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """静态 Bearer 令牌校验；auth.enabled=false 时透明放行。"""
    cfg = request.app.state.settings.auth
    if not cfg.enabled:
        return
    expected = (cfg.token or "").strip()
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    # 常量时间比较：避免逐字符计时侧信道
    if not provided or not expected or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail="未授权：请提供有效的 Bearer 令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )


def memory_or_503(request: Request) -> MemoryRuntime:
    """会话记忆/审计共用同一连接池；未启用时统一 503。"""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(
            status_code=503,
            detail="会话记忆未启用（PG 不可用，或 memory.enabled=false）",
        )
    return memory


def checked_thread_id(thread_id: str | None) -> str:
    if not thread_id or not THREAD_ID_RE.match(thread_id):
        raise HTTPException(
            status_code=422,
            detail=(
                "thread_id 缺失或非法：只允许字母、数字、下划线、连字符（长度 1-64）；"
                "启用会话记忆时必须提供"
            ),
        )
    return thread_id
