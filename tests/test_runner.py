"""Agent Runner 测试。

- 单元：classify_agent_event 把 astream_events v2 原始事件翻译成对外用户事件。
- 端到端：脚本化模型 + mock 数据源走通 问题 → 工具调用 → 结论 → done 的闭环（spec §8.4）。
- 约束：max_seconds 只约束 agent 生产端，不因消费端停顿而误中止；TOOL_STATUS 与工具注册同步。
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from app.agent.runner import TOOL_STATUS, classify_agent_event, stream_chat
from app.agent.tools import build_tools
from app.tools.ocp.mock import MockOcpClient
from app.tools.sql.mock import MockSqlExecutor
from helpers.scripted_model import ScriptedChatModel

# 与 runner.TOOL_STATUS 保持同步：e2e 用例复用同一个工具名与状态文案
_SLOW_SQL_TOOL = "get_slow_sql"
_SLOW_SQL_STATUS = "正在从 OCP 拉取慢SQL…"


def test_classify_tool_start():
    ev = {"event": "on_tool_start", "name": _SLOW_SQL_TOOL, "data": {"input": {"top_n": 5}}}
    assert classify_agent_event(ev) == {"type": "status", "text": _SLOW_SQL_STATUS}


def test_classify_tool_start_unknown_name_falls_back():
    ev = {"event": "on_tool_start", "name": "weird", "data": {}}
    out = classify_agent_event(ev)
    assert out["type"] == "status"
    assert "weird" in out["text"]


def test_classify_final_text_delta():
    class Chunk:
        content = "结论"
        tool_call_chunks = []

    ev = {"event": "on_chat_model_stream", "data": {"chunk": Chunk()}}
    assert classify_agent_event(ev) == {"type": "delta", "text": "结论"}


def test_classify_non_string_content_is_skipped():
    # 真实 AIMessageChunk 的 content 可能是内容块列表（非 str），不应透传
    class Chunk:
        content = [{"type": "text", "text": "结论"}]
        tool_call_chunks = []

    ev = {"event": "on_chat_model_stream", "data": {"chunk": Chunk()}}
    assert classify_agent_event(ev) is None


def test_classify_toolcall_construction_is_skipped():
    class Chunk:
        content = ""
        tool_call_chunks = [{"name": "execute_sql", "args": "{}", "index": 0}]

    ev = {"event": "on_chat_model_stream", "data": {"chunk": Chunk()}}
    assert classify_agent_event(ev) is None


def test_classify_unknown_event_returns_none():
    assert classify_agent_event({"event": "on_random"}) is None


def test_tool_status_matches_tool_registry():
    # TOOL_STATUS 与 build_tools 注册的工具名保持同步，缺了会掉回通用文案
    tools = build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)
    assert set(TOOL_STATUS) == {t.name for t in tools}


@pytest.mark.asyncio
async def test_end_to_end_agent_closed_loop():
    # 第一轮：模型要求调用 get_slow_sql；第二轮：模型给出最终 markdown 结论
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": _SLOW_SQL_TOOL,
                        "args": {"cluster_id": 1, "tenant_id": 1, "limit": 2},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="发现 2 条慢SQL，其中 sq-scan-orders-1 为全表扫描…"),
        ]
    )
    tools = build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)
    events = []
    async for ev in stream_chat(model, tools, [HumanMessage(content="看看有哪些慢SQL并分析")]):
        events.append(ev)

    types = [e["type"] for e in events]
    assert events[0] == {"type": "status", "text": _SLOW_SQL_STATUS}
    assert types[-1] == "done"
    assert "error" not in types
    assert types.index("status") < types.index("delta")
    # 精确相等：同时捕获内容被重复 emit 的问题
    delta_text = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    assert delta_text == "发现 2 条慢SQL，其中 sq-scan-orders-1 为全表扫描…"


class _ReadBackChatModel(ScriptedChatModel):
    """首次模型调用用预设（触发工具），此后从最后一条 ToolMessage 构造回答，验证工具结果真实回传。

    经实测：agent 异步路径在模型 ainvoke 时会走 _astream，且第 2 次调用收到的
    messages 中已包含 ToolMessage（query_slow_sql 的 JSON 返回）。
    """

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            # 首次调用：无工具结果 → 用预设（通常是 tool_call）
            yield ChatGenerationChunk(message=self._chunk_of(self._next_preset()))
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            return
        last = tool_msgs[-1].content
        text = f"工具返回摘要: {str(last)[:200]}"
        yield ChatGenerationChunk(message=AIMessageChunk(content=text))
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))


@pytest.mark.asyncio
async def test_e2e_tool_result_fed_back_to_model():
    model = _ReadBackChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": _SLOW_SQL_TOOL,
                        "args": {"cluster_id": 1, "tenant_id": 1, "limit": 2},
                        "id": "c1",
                    }
                ],
            )
        ]
    )
    tools = build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)
    events = []
    async for ev in stream_chat(model, tools, [HumanMessage(content="慢SQL？")]):
        events.append(ev)

    delta = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    assert delta.startswith("工具返回摘要")  # 走的是 read-back 组合文本，而非罐头预设
    assert "avgElapsedTime" in delta  # 来自真实工具返回（ocp_slow_sqls.json 的字段）
    assert '"ok": true' in delta  # 回传的是工具信封，而不是罐头预设文本


@pytest.mark.asyncio
async def test_slow_consumer_not_cancelled_by_timeout():
    # 消费端在两次事件间停顿超过 max_seconds，不应被 asyncio.timeout 取消（timeout 只约束 agent 生产端）
    model = ScriptedChatModel(responses=[AIMessage(content="hi")])
    tools = build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)
    events = []
    async for ev in stream_chat(model, tools, [HumanMessage(content="x")], max_seconds=0.2):
        events.append(ev)
        await asyncio.sleep(0.4)  # 每步停顿 > 超时：旧实现会在消费端 await 处抛 CancelledError
    assert events[-1]["type"] in ("done", "error")
    assert "error" not in [e["type"] for e in events]


@pytest.mark.asyncio
async def test_e2e_explain_tool_event_carries_plan_view():
    """执行计划可视化：tool 事件必须带归一化后的 plan（先序 depth + 算子汇总），供前端画树。"""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_sql_explain",
                        "args": {"cluster_id": 1, "tenant_id": 1, "uid": "plan-8f3a1c02"},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="orders 是全表扫描。"),
        ]
    )
    tools = build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)
    events = []
    async for ev in stream_chat(model, tools, [HumanMessage(content="看下执行计划")]):
        events.append(ev)

    tool_events = [e for e in events if e["type"] == "tool"]
    assert len(tool_events) == 1
    tool_ev = tool_events[0]
    assert tool_ev["ok"] is True and tool_ev["name"] == "get_sql_explain"
    plan = tool_ev["plan"]
    # uid 来自工具参数；真实 OCP 报文不带 sqlId（那是 top_plan 的字段）
    assert plan["uid"] == "plan-8f3a1c02" and plan["sql_id"] == ""
    assert plan["node_count"] == 10
    assert [(n["operator"], n["depth"]) for n in plan["nodes"]][:3] == [
        ("PHY_SCALAR_AGGREGATE", 0),
        ("PHY_HASH_JOIN", 1),
        ("PHY_MERGE_JOIN", 2),
    ]
    assert plan["nodes"][4]["name"] == "ER" and plan["nodes"][4]["cost"] == 6
    assert plan["summary"][0]["operator"] == "PHY_SCALAR_AGGREGATE"
    assert plan["truncated"] is False


@pytest.mark.asyncio
async def test_e2e_other_tools_have_no_plan_key():
    """其余工具不占位：事件里没有 plan 键，前端不会渲染空计划块。"""
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": _SLOW_SQL_TOOL, "args": {"cluster_id": 1, "tenant_id": 1, "limit": 2}, "id": "c1"}
                ],
            ),
            AIMessage(content="没有慢SQL。"),
        ]
    )
    tools = build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)
    events = []
    async for ev in stream_chat(model, tools, [HumanMessage(content="慢SQL？")]):
        events.append(ev)

    tool_events = [e for e in events if e["type"] == "tool"]
    assert tool_events and all("plan" not in e for e in tool_events)
