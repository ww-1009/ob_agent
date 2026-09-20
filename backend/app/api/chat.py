"""聊天与健康检查路由。"""
from __future__ import annotations

from typing import AsyncIterator, List, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent.confirm import ConfirmationBroker
from app.agent.runner import stream_chat
from app.sse import sse_frame

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)


def _to_langchain(messages: List[ChatMessage]) -> List[BaseMessage]:
    """转换历史消息为 langchain 消息（仅 user/assistant）。"""
    return [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in messages
    ]


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="LLM 未配置：请在 config.yaml / .env 填写 llm.base_url、api_key、model",
        )

    async def gen() -> AsyncIterator[str]:
        # 每次请求独立 broker：owner 注册为 producer task（stream_chat 内部），
        # 其 finally 负责 fail_all——客户端断连/流结束都不会留下挂起的确认。
        broker = ConfirmationBroker()
        state.active_broker = broker
        try:
            async for ev in stream_chat(
                state.model,
                state.tools,
                _to_langchain(req.messages),
                max_seconds=state.settings.agent.max_seconds,
                recursion_limit=state.settings.agent.recursion_limit,
                broker=broker,
                confirm_enabled=state.settings.agent.confirm_db_ops,
                confirm_timeout_seconds=state.settings.agent.confirm_timeout_seconds,
            ):
                yield sse_frame(ev)
        finally:
            if getattr(state, "active_broker", None) is broker:
                state.active_broker = None

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
    }
