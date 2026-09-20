"""人工确认通道：数据库操作执行前等待用户在前端批准（HITL）。

组件分层：
- ConfirmationBroker：request_id → 确认 Future 的登记表；发起 SSE 流的 task 注册为 owner，
  流断开/结束时 fail_all 统一释放仍在等待的确认（拒绝语义），杜绝孤儿等待。
- build_confirm_middleware：AgentMiddleware.awrap_tool_call 拦截点；命中 DB 工具且 broker 在场时
  先发事件、再等 Future；批准 → handler 执行真实工具；拒绝/超时/断连 → 返回拒绝观察 ToolMessage。

事件形态（经 runner 的 emit 通道进入 SSE 流）：
  {type:"confirm_request", request_id, tool, tool_label, args, expires_in}
  {type:"confirm_timeout", request_id}
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# 需要用户确认的“真正访问数据库”工具：仅 execute_sql（直连租户执行只读 SQL）。
# 其余 OCP 只读元数据/数据查询（get_slow_sql、get_sql_explain 等）与文件读工具
# 不纳入人工审批，保持既有行为。
CONFIRM_TOOL_LABELS: dict[str, str] = {
    "execute_sql": "执行只读 SQL 查询",
}


@dataclass
class _Pending:
    future: "asyncio.Future[bool]"
    owner: "asyncio.Task | None" = None


class ConfirmationBroker:
    """确认登记表：request_id → Future；owner 流断开时统一 fail_all。"""

    def __init__(self) -> None:
        self._pending: dict[str, _Pending] = {}

    def register(self, request_id: str, *, owner: "asyncio.Task | None" = None) -> "asyncio.Future[bool]":
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[bool]" = loop.create_future()
        self._pending[request_id] = _Pending(future=fut, owner=owner)
        return fut

    def resolve(self, request_id: str, approved: bool) -> bool:
        """投递用户答复。未知 id → False（路由转 404）；已答复 → False（路由转 409）。"""
        entry = self._pending.pop(request_id, None)
        if entry is None:
            return False
        if entry.future.done():
            return False
        entry.future.set_result(bool(approved))
        return True

    def discard(self, request_id: str) -> None:
        self._pending.pop(request_id, None)

    def owner_request_ids(self, owner: "asyncio.Task") -> list:
        return [rid for rid, e in self._pending.items() if e.owner is owner]

    def fail_all(self, owner: "asyncio.Task | None" = None, *, reason: str = "stream_closed") -> list:
        """以拒绝语义释放等待中的确认（owner=None 时全部）。返回被释放的 request_id 列表。"""
        released = []
        for rid in list(self._pending):
            entry = self._pending.get(rid)
            if entry is None:
                continue
            if owner is not None and entry.owner is not owner:
                continue
            self._pending.pop(rid, None)
            if not entry.future.done():
                entry.future.set_result(False)
            released.append(rid)
        if released:
            logger.info("确认等待被释放(%s): %s", reason, released)
        return released

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def _emit(emit: Optional[Callable[[dict], object]], ev: dict) -> None:
    if emit is None:
        return
    out = emit(ev)
    if inspect.isawaitable(out):
        asyncio.ensure_future(out)


def build_confirm_middleware(
    *,
    emit: Optional[Callable[[dict], object]] = None,
    broker: Optional[ConfirmationBroker] = None,
    owner: "asyncio.Task | None" = None,
    timeout_seconds: float = 120,
    tool_names: Optional[dict[str, str]] = None,
) -> AgentMiddleware:
    """构造 DB 操作确认中间件。broker=None 时中间件透明直通（保持既有行为）。"""
    labels = tool_names if tool_names is not None else CONFIRM_TOOL_LABELS

    class ConfirmMiddleware(AgentMiddleware):
        name = "db_op_confirmation"

        async def awrap_tool_call(self, request, handler):
            call = request.tool_call or {}
            name = call.get("name") or ""
            if broker is None or name not in labels:
                return await handler(request)

            rid = f"cf_{uuid.uuid4().hex[:16]}"
            args = call.get("args") or {}
            fut = broker.register(rid, owner=owner)
            _emit(emit, {
                "type": "confirm_request",
                "request_id": rid,
                "tool": name,
                "tool_label": labels[name],
                "args": args,
                "expires_in": int(timeout_seconds),
            })
            approved = False
            timed_out = False
            try:
                approved = await asyncio.wait_for(fut, timeout=timeout_seconds)
            except (TimeoutError, asyncio.TimeoutError):
                timed_out = True
                broker.discard(rid)
            except asyncio.CancelledError:
                broker.discard(rid)
                raise
            if timed_out:
                _emit(emit, {"type": "confirm_timeout", "request_id": rid})

            if not approved:
                reason = "确认超时，已默认拒绝" if timed_out else "用户拒绝了该操作"
                return ToolMessage(
                    content=f"__confirm_denied__: {reason}。不要重复尝试同一操作，请换只读途径或向用户说明。",
                    name=name,
                    tool_call_id=call.get("id") or "",
                    status="error",
                )
            return await handler(request)

    return ConfirmMiddleware()
