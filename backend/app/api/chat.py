"""聊天与健康检查路由。"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, List, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent.runner import stream_chat
from app.api.deps import checked_thread_id
from app.sse import sse_frame

logger = logging.getLogger(__name__)

router = APIRouter()
# 健康检查单独成 router：探活需要免令牌，故在 main 中以不同依赖挂载
health_router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    # 启用会话记忆时必填（langgraph 检查点的会话键）；关闭记忆时可省略
    thread_id: str | None = None


def _to_langchain(messages: List[ChatMessage]) -> List[BaseMessage]:
    """转换历史消息为 langchain 消息（仅 user/assistant）。"""
    return [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in messages
    ]


# 进程内保留的 thread 锁数量上限。锁与 thread_id 一一对应且不随会话删除清理，
# 无上限时映射会随不同 thread_id 数量无界增长。
_MAX_TRACKED_THREAD_LOCKS = 1024


def _thread_lock(state, thread_id: str) -> asyncio.Lock:
    """按会话取互斥锁。

    同一 thread 并发两轮会让 langgraph 写出分叉的检查点（上下文互相覆盖），
    因此显式串行化：占用中直接 409，而不是静默排队或写坏状态。

    锁对象在进程内按 thread_id 复用，并在超过 _MAX_TRACKED_THREAD_LOCKS 时驱逐
    **未被占用**的旧锁：持有中的锁绝不驱逐，而 asyncio.Lock.acquire() 对空闲锁不挂起、
    调用方取锁与检查之间没有 await，因此驱逐不会让同一 thread 同时出现两把活锁。
    """
    locks = getattr(state, "thread_locks", None)
    if locks is None:
        locks = {}
        state.thread_locks = locks
    lock = locks.get(thread_id)
    if lock is not None:
        return lock
    if len(locks) >= _MAX_TRACKED_THREAD_LOCKS:
        # 先驱逐旧锁再插入新锁：新锁绝不在本轮被回收
        for tid in list(locks):
            if len(locks) < _MAX_TRACKED_THREAD_LOCKS:
                break
            if not locks[tid].locked():
                del locks[tid]
    lock = asyncio.Lock()
    locks[thread_id] = lock
    return lock


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置：请在 config.yaml / .env 填写 llm.base_url、api_key、model",
        )

    # HITL fail closed：开了人工确认却没有确认通道时，宁可报 503 也不能无审批执行受控工具
    broker = getattr(state, "confirm_broker", None)
    if state.settings.agent.confirm_db_ops and broker is None:
        raise HTTPException(
            status_code=503,
            detail="确认通道不可用：启用人工审批时无法开始对话",
        )

    memory = getattr(state, "memory", None)
    user_text: str | None = None
    if memory is None:
        # 无记忆：沿用无状态语义，messages 须含完整历史
        thread_id: str | None = None
        history = _to_langchain(req.messages)
    else:
        # 有记忆：只认本轮新消息，历史由检查点按 thread_id 提供
        thread_id = checked_thread_id(req.thread_id)
        last = req.messages[-1]
        if last.role != "user":
            raise HTTPException(
                status_code=422,
                detail="启用会话记忆时只传本轮新消息，messages 最后一条必须是 user",
            )
        user_text = last.content
        history = [HumanMessage(content=user_text)]

    # 同一会话串行化：占用中直接拒绝，避免并发两轮写坏检查点分支。
    # 在端点内取锁（asyncio 单线程下 locked() 检查与 acquire 之间不会让出），
    # 在流的 finally 释放，保证客户端断连时也能解锁。
    thread_lock: asyncio.Lock | None = None
    if thread_id is not None:
        thread_lock = _thread_lock(state, thread_id)
        if thread_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="该会话正在处理上一条消息，请等待完成，或新建会话后再提问",
            )
        await thread_lock.acquire()

    released = False

    def release_lock() -> None:
        nonlocal released
        if thread_lock is not None and not released:
            released = True
            thread_lock.release()

    async def gen() -> AsyncIterator[str]:
        # 进程级共享 broker：request_id 全局唯一，寻址不依赖"最近一次请求"这类可变槽位；
        # 跨会话隔离靠 owner（stream_chat 把 owner 注册为自身的 producer task），
        # 其 finally 的 fail_all(owner) 保证客户端断连/流结束都不会留下挂起的确认。
        parts: list[str] = []
        persist = memory is not None

        async def persist_answer() -> None:
            """落库助手终稿（幂等：写过即置 persist=False，避免重复）。"""
            nonlocal persist
            if not persist:
                return
            persist = False
            answer = "".join(parts)
            if not answer.strip():
                return
            try:
                await memory.store.append(thread_id, "assistant", answer)
            except Exception as e:  # noqa: BLE001 - 落库失败不应影响对话
                logger.warning("写入助手回答失败：%s", e)

        async def audit_tool(ev: dict) -> None:
            """工具轨迹落库（审计）。失败只记日志，不影响对话。"""
            if memory is None:
                return
            args = ev.get("args")
            try:
                await memory.audit.append(
                    thread_id=thread_id,
                    tool=str(ev.get("name") or ""),
                    args=args if isinstance(args, dict) else {},
                    ok=bool(ev.get("ok")),
                    error=ev.get("error"),
                    rows=ev.get("rows"),
                    truncated=ev.get("truncated"),
                    approved=ev.get("approved"),
                    duration_ms=ev.get("duration_ms"),
                )
            except Exception as e:  # noqa: BLE001 - 审计写入失败不应打断对话
                logger.warning("写入审计失败：%s", e)

        if persist:
            # 先落 user 消息：即便随后报错/中断，历史里也留有这次提问
            try:
                await memory.store.append(thread_id, "user", user_text or "")
            except Exception as e:  # noqa: BLE001
                logger.warning("写入用户消息失败，本次不落库：%s", e)
                persist = False
        try:
            async for ev in stream_chat(
                state.model,
                state.tools,
                history,
                thread_id=thread_id,
                checkpointer=memory.checkpointer if memory is not None else None,
                max_seconds=state.settings.agent.max_seconds,
                recursion_limit=state.settings.agent.recursion_limit,
                broker=broker,
                confirm_enabled=state.settings.agent.confirm_db_ops,
                confirm_timeout_seconds=state.settings.agent.confirm_timeout_seconds,
            ):
                if ev.get("type") == "tool" and ev.get("phase") == "end":
                    await audit_tool(ev)
                if ev.get("type") == "delta":
                    parts.append(str(ev.get("text") or ""))
                elif ev.get("type") == "done":
                    # 先落库再发 done：前端收到 done 后立即刷新列表即可看到完整计数
                    await persist_answer()
                yield sse_frame(ev)
        finally:
            # 错误/客户端中断路径：把已收到的部分回答落库（空则跳过）
            await persist_answer()
            release_lock()

    try:
        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except BaseException:
        # 极端情况下流未被消费：确保锁不泄漏
        release_lock()
        raise


@health_router.get("/api/health")
async def health(request: Request):
    st = request.app.state.settings
    memory = getattr(request.app.state, "memory", None)
    error = getattr(request.app.state, "memory_error", None)
    payload = {
        "status": "ok",
        "ocp_provider": st.ocp.provider,
        "sql_provider": st.sql_ro.provider,
        "llm_configured": st.llm.is_configured,
        "memory_enabled": memory is not None,
        "auth_enabled": st.auth.enabled,
    }
    # 仅在实际降级时暴露原因，便于线上判断"为什么没有记忆/审计"
    if memory is None and error:
        payload["memory_error"] = error
    return payload
