"""确认答复路由：POST /api/chat/confirm（独立 APIRouter，create_app 装配）。

SSE 是单向流，用户的「允许/拒绝」经此反向通道投递到 ConfirmationBroker 中挂起的 Future。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


class ConfirmDecision(BaseModel):
    request_id: str
    approved: bool


def make_confirm_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat/confirm")
    async def confirm(req: ConfirmDecision, request: Request):
        state = request.app.state
        broker = getattr(state, "confirm_broker_factory", None)
        # 每请求 broker 挂在 active_broker；全局 broker 用于测试注入/静态模式
        broker = getattr(state, "active_broker", None) or getattr(state, "confirm_broker", None)
        if broker is None:
            raise HTTPException(status_code=503, detail="确认通道不可用")
        if broker.resolve(req.request_id, req.approved):
            return {"ok": True}
        raise HTTPException(status_code=404, detail="确认请求不存在或已过期")

    return router
