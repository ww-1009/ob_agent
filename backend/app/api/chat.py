"""聊天与健康检查路由。"""
from __future__ import annotations

import logging
import re
from typing import AsyncIterator, List, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent.confirm import ConfirmationBroker
from app.agent.runner import stream_chat
from app.sse import sse_frame

logger = logging.getLogger(__name__)

router = APIRouter()

# 会话 ID 白名单：同时用作 chat_message.thread_id 与检查点 thread_id
THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


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


def _checked_thread_id(thread_id: str | None) -> str:
    if not thread_id or not THREAD_ID_RE.match(thread_id):
        raise HTTPException(
            status_code=422,
            detail="启用会话记忆时 thread_id 必填，且只允许字母、数字、下划线、连字符（长度 1-64）",
        )
    return thread_id


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置：请在 config.yaml / .env 填写 llm.base_url、api_key、model",
        )

    memory = getattr(state, "memory", None)
    user_text: str | None = None
    if memory is None:
        # 无记忆：沿用无状态语义，messages 须含完整历史
        thread_id: str | None = None
        history = _to_langchain(req.messages)
    else:
        # 有记忆：只认本轮新消息，历史由检查点按 thread_id 提供
        thread_id = _checked_thread_id(req.thread_id)
        last = req.messages[-1]
        if last.role != "user":
            raise HTTPException(
                status_code=422,
                detail="启用会话记忆时只传本轮新消息，messages 最后一条必须是 user",
            )
        user_text = last.content
        history = [HumanMessage(content=user_text)]

    async def gen() -> AsyncIterator[str]:
        # 每次请求独立 broker：owner 注册为 producer task（stream_chat 内部），
        # 其 finally 负责 fail_all——客户端断连/流结束都不会留下挂起的确认。
        broker = ConfirmationBroker()
        state.active_broker = broker
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
                if ev.get("type") == "delta":
                    parts.append(str(ev.get("text") or ""))
                elif ev.get("type") == "done":
                    # 先落库再发 done：前端收到 done 后立即刷新列表即可看到完整计数
                    await persist_answer()
                yield sse_frame(ev)
        finally:
            if getattr(state, "active_broker", None) is broker:
                state.active_broker = None
            # 错误/客户端中断路径：把已收到的部分回答落库（空则跳过）
            await persist_answer()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/health")
async def health(request: Request):
    st = request.app.state.settings
    return {
        "status": "ok",
        "ocp_provider": st.ocp.provider,
        "sql_provider": st.meta_db.provider,
        "llm_configured": st.llm.is_configured,
        "memory_enabled": getattr(request.app.state, "memory", None) is not None,
    }
