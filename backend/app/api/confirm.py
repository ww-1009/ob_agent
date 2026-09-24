"""确认答复路由：POST /api/chat/confirm（独立 APIRouter，create_app 装配）。

SSE 是单向流，用户的「允许/拒绝」经此反向通道投递到 ConfirmationBroker 中挂起的 Future。

broker 是**进程级共享**的（app.state.confirm_broker，见 main.create_app）：request_id 为
全局唯一的 uuid，因此不需要「最近一次请求」这类寻址槽位；并发会话之间的隔离由 runner
按 owner=producer task 执行的 fail_all 负责。
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
        broker = getattr(state, "confirm_broker", None)
        if broker is None:
            raise HTTPException(status_code=503, detail="确认通道不可用")
        if broker.resolve(req.request_id, req.approved):
            return {"ok": True}
        raise HTTPException(status_code=404, detail="确认请求不存在或已过期")

    return router
