"""只读工具调用的审计查询接口。

写入口在 api/chat.py 的 SSE 循环里（收到工具轨迹事件时落库），这里只负责查询。
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.agent.tool_labels import tool_label
from app.api.deps import checked_thread_id, memory_or_503


def make_audit_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/audit")
    async def list_audit(
        request: Request,
        limit: int = Query(default=0, ge=0, le=500),
        thread_id: str | None = None,
        tool: str | None = None,
    ):
        memory = memory_or_503(request)
        tid = checked_thread_id(thread_id) if thread_id else None
        n = limit or 100
        events = await memory.audit.list_events(n, thread_id=tid, tool=tool)
        return {
            "items": [
                {
                    "id": e.id,
                    "thread_id": e.thread_id,
                    "tool": e.tool,
                    "label": tool_label(e.tool),
                    "args": e.args,
                    "ok": e.ok,
                    "error": e.error,
                    "rows": e.rows,
                    "truncated": e.truncated,
                    "approved": e.approved,
                    "duration_ms": e.duration_ms,
                    "created_at": e.created_at.isoformat(),
                }
                for e in events
            ]
        }

    return router
