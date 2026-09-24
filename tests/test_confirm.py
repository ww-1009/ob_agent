"""ConfirmationBroker 单测 + 确认中间件行为（HITL）。"""
import asyncio

import pytest
from langchain_core.messages import ToolMessage

from app.agent.confirm import (
    CONFIRM_TOOL_LABELS,
    ConfirmationBroker,
    build_confirm_middleware,
)


@pytest.mark.asyncio
async def test_register_resolve_roundtrip():
    broker = ConfirmationBroker()
    fut = broker.register("r1")
    assert broker.pending_count == 1
    assert broker.resolve("r1", True) is True
    assert await asyncio.wait_for(fut, timeout=1) is True
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_resolve_unknown_id_is_false():
    broker = ConfirmationBroker()
    assert broker.resolve("nope", True) is False


@pytest.mark.asyncio
async def test_resolve_twice_second_false():
    broker = ConfirmationBroker()
    broker.register("r1")
    assert broker.resolve("r1", True) is True
    assert broker.resolve("r1", True) is False


@pytest.mark.asyncio
async def test_fail_all_rejects_owned_only():
    broker = ConfirmationBroker()
    owner_a = asyncio.current_task()

    async def other():
        return asyncio.current_task()
    owner_b = await asyncio.create_task(other())

    fa = broker.register("a", owner=owner_a)
    fb = broker.register("b", owner=owner_b)
    released = broker.fail_all(owner_a)
    assert released == ["a"]
    assert await asyncio.wait_for(fa, timeout=1) is False  # 拒绝语义
    assert broker.pending_count == 1  # b 仍在等自己的 owner
    broker.fail_all(owner_b)
    assert await asyncio.wait_for(fb, timeout=1) is False
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_fail_all_without_owner_releases_everything():
    broker = ConfirmationBroker()
    f1 = broker.register("x")
    f2 = broker.register("y")
    assert set(broker.fail_all()) == {"x", "y"}
    assert (await asyncio.wait_for(f1, timeout=1)) is False
    assert (await asyncio.wait_for(f2, timeout=1)) is False


@pytest.mark.asyncio
async def test_pending_ids_by_owner():
    broker = ConfirmationBroker()
    task = asyncio.current_task()
    broker.register("p1", owner=task)
    assert broker.owner_request_ids(task) == ["p1"]


class _Req:
    def __init__(self, name, args, tc_id="tc-1"):
        self.tool_call = {"name": name, "args": args, "id": tc_id}


async def _handler(request):
    return ToolMessage(content="executed", name=request.tool_call["name"], tool_call_id=request.tool_call["id"])


@pytest.mark.asyncio
async def test_middleware_passes_through_non_db_tool():
    mw = build_confirm_middleware(broker=ConfirmationBroker())
    out = await mw.awrap_tool_call(_Req("get_tenant_info", {}), _handler)
    assert out.content == "executed"


@pytest.mark.asyncio
async def test_middleware_approve_executes_tool():
    broker = ConfirmationBroker()
    events = []

    async def emit(ev):
        events.append(ev)

    mw = build_confirm_middleware(emit=emit, broker=broker, timeout_seconds=5)
    req = _Req("execute_sql", {"cluster_name": "c1", "tenant_name": "t1", "db_name": "d", "sql": "select 1", "tenant_type": "MYSQL"})

    async def resolve_soon():
        for _ in range(200):
            await asyncio.sleep(0.005)
            if broker._pending and broker.resolve(next(iter(broker._pending)), True):
                return

    out, _ = await asyncio.gather(mw.awrap_tool_call(req, _handler), resolve_soon())
    assert out.content == "executed"
    assert events[0]["type"] == "confirm_request"
    assert events[0]["tool"] == "execute_sql"
    assert events[0]["args"]["sql"] == "select 1"
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_middleware_deny_short_circuits_handler():
    broker = ConfirmationBroker()
    events = []

    async def emit(ev):
        events.append(ev)

    mw = build_confirm_middleware(emit=emit, broker=broker, timeout_seconds=5)
    req = _Req("execute_sql", {"sql": "select 1"})

    async def deny_soon():
        for _ in range(200):
            await asyncio.sleep(0.005)
            if broker._pending and broker.resolve(next(iter(broker._pending)), False):
                return

    out, _ = await asyncio.gather(mw.awrap_tool_call(req, _handler), deny_soon())
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "__confirm_denied__" in out.content
    assert "executed" not in out.content  # handler 未被调用


@pytest.mark.asyncio
async def test_middleware_timeout_denies_and_emits_timeout_event():
    broker = ConfirmationBroker()
    events = []

    async def emit(ev):
        events.append(ev)

    mw = build_confirm_middleware(emit=emit, broker=broker, timeout_seconds=0.1)
    out = await mw.awrap_tool_call(_Req("execute_sql", {"sql": "select 1"}), _handler)
    await asyncio.sleep(0)  # _emit 对 async emit 只 ensure_future，让调度的事件任务先跑完
    assert out.status == "error"
    assert "超时" in out.content
    # confirm_timeout 之后还会补发一条被拒绝的 tool_trace 事件，故按类型查找而非取最后一个
    timeout_evs = [e for e in events if e["type"] == "confirm_timeout"]
    assert timeout_evs == [{"type": "confirm_timeout", "request_id": events[0]["request_id"]}]
    assert broker.pending_count == 0


@pytest.mark.asyncio
async def test_middleware_transparent_without_broker():
    mw = build_confirm_middleware(broker=None)
    out = await mw.awrap_tool_call(_Req("execute_sql", {"sql": "select 1"}), _handler)
    assert out.content == "executed"  # 无 broker：不拦截


def test_confirm_labels_only_gate_execute_sql():
    # 当前设计：仅直连租户执行 SQL 的 execute_sql 需要人工确认；
    # 其余 OCP 只读查询与文件工具不拦截（见 confirm.py:30-35 注释）
    assert set(CONFIRM_TOOL_LABELS) == {"execute_sql"}
