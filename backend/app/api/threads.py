"""会话历史路由：列出会话 / 读取历史 / 删除会话。

仅走 LangGraph 检查点的 async 接口（AsyncPostgresSaver 的同步 list()/get_tuple()
在事件循环线程调用会抛 InvalidStateError）。
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.api.deps import checked_thread_id as _checked_thread_id, memory_or_503 as _memory_or_503


def make_threads_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/threads")
    async def list_threads(
        request: Request,
        limit: int = Query(default=0, ge=0, le=500),
    ):
        memory = _memory_or_503(request)
        n = limit or request.app.state.settings.memory.list_limit
        items = await memory.store.list_threads(n)
        return {
            "items": [
                {
                    "thread_id": t.thread_id,
                    "title": t.title,
                    "created_at": t.created_at.isoformat(),
                    "updated_at": t.updated_at.isoformat(),
                    "message_count": t.message_count,
                }
                for t in items
            ]
        }

    @router.get("/api/threads/{thread_id}/messages")
    async def thread_messages(
        thread_id: str,
        request: Request,
        limit: int = Query(default=0, ge=0, le=2000),
    ):
        memory = _memory_or_503(request)
        tid = _checked_thread_id(thread_id)
        n = limit or request.app.state.settings.memory.messages_limit
        messages = await memory.store.get_messages(tid, n)
        return {
            "thread_id": tid,
            "items": [
                {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
                for m in messages
            ],
        }

    @router.delete("/api/threads/{thread_id}")
    async def delete_thread(thread_id: str, request: Request):
        memory = _memory_or_503(request)
        tid = _checked_thread_id(thread_id)
        deleted = await memory.adelete_thread(tid)
        return {"ok": True, "deleted_messages": deleted}

    return router
