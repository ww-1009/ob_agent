"""HITL 确认通道的可达性回归测试。

历史缺陷：`/api/chat/confirm` 只从进程内单一可变槽位 `state.active_broker` 取 broker，
于是

- 单 worker：并发会话互相顶掉，先启动的流审批 404；后启动的流结束时还会把槽清空，
  使仍在等审批的流变成 503；
- 前端（useChat.js）把 404/409/503 当作"已答复"静默关闭卡片，用户在超时后被自动拒绝。

现在 broker 是进程级共享的 `app.state.confirm_broker`，按全局唯一的 request_id 寻址；
跨会话隔离由 runner 注册的 `owner=producer_task` + `fail_all(owner)` 负责。
"""
from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator, List, Optional

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.api.chat import _MAX_TRACKED_THREAD_LOCKS, _thread_lock
from app.config import load_settings
from app.main import create_app

# 指向必然不存在的配置文件，避免读到开发者本机 backend/config.yaml 的真实 LLM 配置
_MISSING_CONFIG = "/nonexistent-ob-agent-test-config.yaml"

_SQL_TOOL_ARGS = {
    "tenant_name": "t1",
    "cluster_name": "c1",
    "db_name": "dckkdb",
    "sql": "select * from orders",
    "tenant_type": "MYSQL",
}


class LastMessageHitlModel(BaseChatModel):
    """无状态模型桩：最后一条是 HumanMessage 就调 execute_sql（触发人工确认），否则收尾。

    相比 SingleCounter 的 ScriptedChatModel，这里必须无状态：并发两条流各自独立推进，
    共享计数器会把两条流的脚本串在一起。
    """

    @property
    def _llm_type(self) -> str:
        return "hitl-stub"

    def bind_tools(self, tools, **kwargs):  # 忽略真实工具绑定：行为完全由本桩决定
        return self

    @staticmethod
    def _next(messages: List[BaseMessage]) -> AIMessage:
        if messages and isinstance(messages[-1], HumanMessage):
            return AIMessage(
                content="",
                tool_calls=[{"name": "execute_sql", "args": dict(_SQL_TOOL_ARGS), "id": "c1"}],
            )
        return AIMessage(content="完成")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(list(messages)))])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        msg = self._next(list(messages))
        if msg.tool_calls:
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": tc["name"],
                        "args": json.dumps(tc["args"], ensure_ascii=False),
                        "id": tc["id"],
                        "index": i,
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ],
            )
        else:
            chunk = AIMessageChunk(content=msg.content or "")
        yield ChatGenerationChunk(message=chunk)
        # 末尾空 chunk 触发流式 tool_call 的 finalize（缺失可能不执行工具）
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))


def _app(*, confirm_db_ops: bool = True, model=None):
    settings = load_settings(
        config_path=_MISSING_CONFIG,
        env={"OCP_PROVIDER": "mock", "SQL_PROVIDER": "mock"},
    )
    settings.agent.confirm_db_ops = confirm_db_ops
    settings.agent.confirm_timeout_seconds = 10
    settings.agent.max_seconds = 20
    return create_app(settings, model=model if model is not None else LastMessageHitlModel())


class SseStream:
    """在后台线程消费一条真实 SSE 流；主线程按事件类型等待。

    必须走真实 uvicorn：starlette 的 TestClient 会在一个 portal 调用里把 ASGI 应用
    await 到底并把 body 攒进 BytesIO，因此**无法**与"流仍在等待审批"的请求并发交互，
    用它测 HITL 只会得到"审批到达时请求已超时"的假失败。
    """

    def __init__(self, client: httpx.Client, url: str, body: dict) -> None:
        self._cm = client.stream("POST", url, json=body)
        self.resp = self._cm.__enter__()
        self.status = self.resp.status_code
        self.events: list[dict] = []
        self._cv = threading.Condition()
        self._finished = False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            for line in self.resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                with self._cv:
                    self.events.append(ev)
                    self._cv.notify_all()
        except Exception:  # noqa: BLE001 - teardown 关流时读到已关闭的 fd 属预期，不该炸线程
            pass
        finally:
            with self._cv:
                self._finished = True
                self._cv.notify_all()

    def wait_for(self, etype: str, timeout: float = 15.0) -> dict:
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                for ev in self.events:
                    if ev.get("type") == etype:
                        return ev
                left = deadline - time.monotonic()
                if left <= 0 or (self._finished and not self._thread.is_alive()):
                    raise AssertionError(
                        f"等待事件 {etype} 超时；已收到 {[e.get('type') for e in self.events]}"
                    )
                self._cv.wait(min(left, 0.5))

    def types(self) -> list[str]:
        with self._cv:
            return [str(e.get("type")) for e in self.events]

    def close(self) -> None:
        try:
            self._cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001 - 关闭失败不影响断言结果
            pass


def _tool_end_events(stream: SseStream) -> list[dict]:
    return [e for e in stream.events if e.get("type") == "tool" and e.get("phase") == "end"]


@contextlib.contextmanager
def _serve(app) -> Iterator[str]:
    """在后台线程起一个真实 uvicorn（随机端口），返回 base_url。"""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn 未能在 15s 内启动")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_live_stream_confirm_roundtrip():
    """流停在审批上时，确认请求必须能在同一进程内被寻址到（核心回归）。"""
    app = _app()
    broker_before = app.state.confirm_broker
    with _serve(app) as base, httpx.Client(timeout=30) as client:
        stream = SseStream(client, f"{base}/api/chat", {"messages": [{"role": "user", "content": "查一下"}]})
        try:
            assert stream.status == 200
            req = stream.wait_for("confirm_request")
            assert req["tool"] == "execute_sql"
            assert req["request_id"]

            r = client.post(
                f"{base}/api/chat/confirm",
                json={"request_id": req["request_id"], "approved": True},
            )
            assert r.status_code == 200, r.text
            assert r.json() == {"ok": True}

            stream.wait_for("done")
            assert "error" not in stream.types()
            ends = _tool_end_events(stream)
            assert ends and ends[0].get("ok") is True
        finally:
            stream.close()

    # 流结束后共享 broker 不被改写或清空（旧实现会在这里把槽位置空）
    assert app.state.confirm_broker is broker_before
    assert not hasattr(app.state, "active_broker")


def test_two_concurrent_streams_both_approvable():
    """两条并发流各自的审批都必须可达（旧实现：后启动的流顶掉先启动的流 → 404）。"""
    app = _app()
    with _serve(app) as base, httpx.Client(timeout=30) as client:
        first = SseStream(client, f"{base}/api/chat", {"messages": [{"role": "user", "content": "第一个问题"}]})
        second = SseStream(client, f"{base}/api/chat", {"messages": [{"role": "user", "content": "第二个问题"}]})
        try:
            first_req = first.wait_for("confirm_request")
            second_req = second.wait_for("confirm_request")
            assert first_req["request_id"] != second_req["request_id"]

            # 先答复先启动的流：旧实现在这一步会因为槽位被第二条流覆盖而 404
            r1 = client.post(
                f"{base}/api/chat/confirm",
                json={"request_id": first_req["request_id"], "approved": True},
            )
            assert r1.status_code == 200, r1.text

            r2 = client.post(
                f"{base}/api/chat/confirm",
                json={"request_id": second_req["request_id"], "approved": True},
            )
            assert r2.status_code == 200, r2.text

            first.wait_for("done")
            second.wait_for("done")
            assert "error" not in first.types()
            assert "error" not in second.types()
        finally:
            first.close()
            second.close()


def test_confirm_unknown_id_404():
    app = _app()
    client = TestClient(app)
    r = client.post("/api/chat/confirm", json={"request_id": "cf_nope", "approved": True})
    assert r.status_code == 404


def test_confirm_without_broker_503():
    app = _app()
    del app.state.confirm_broker
    client = TestClient(app)
    r = client.post("/api/chat/confirm", json={"request_id": "cf_nope", "approved": True})
    assert r.status_code == 503


def test_chat_503_when_hitl_enabled_without_broker():
    """fail closed：开了人工确认却没有确认通道时，不能无审批执行受控工具。"""
    app = _app(confirm_db_ops=True)
    del app.state.confirm_broker
    client = TestClient(app)
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "查一下"}]})
    assert r.status_code == 503
    assert "确认通道" in r.json()["detail"]


def test_chat_still_works_without_broker_when_hitl_disabled():
    app = _app(confirm_db_ops=False)
    del app.state.confirm_broker
    client = TestClient(app)
    with client.stream("POST", "/api/chat", json={"messages": [{"role": "user", "content": "查一下"}]}) as resp:
        assert resp.status_code == 200
        types = [
            json.loads(ln[6:])["type"]
            for ln in resp.iter_lines()
            if ln.startswith("data: ")
        ]
    assert "done" in types
    assert "error" not in types


def test_thread_lock_map_is_bounded():
    app = _app()
    state = app.state
    for i in range(_MAX_TRACKED_THREAD_LOCKS * 2):
        _thread_lock(state, f"t{i}")
    assert len(state.thread_locks) <= _MAX_TRACKED_THREAD_LOCKS


@pytest.mark.asyncio
async def test_thread_lock_eviction_never_drops_a_held_lock():
    state = SimpleNamespace()
    held = _thread_lock(state, "held")
    await held.acquire()
    try:
        for i in range(_MAX_TRACKED_THREAD_LOCKS + 10):
            _thread_lock(state, f"t{i}")
        assert state.thread_locks["held"] is held
        assert len(state.thread_locks) <= _MAX_TRACKED_THREAD_LOCKS + 1
    finally:
        held.release()