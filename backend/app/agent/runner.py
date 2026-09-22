"""Agent 编排：把 langchain agent 的 astream_events 翻译成对外事件流。

对外事件（dict）：{type: "status"|"delta"|"error"|"done", ...}

- stream_chat 对历史消息运行 create_agent 驱动的 agent，逐个 yield 用户事件。
- classify_agent_event 是纯函数：单条 v2 原始事件 → 用户事件 / None。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Dict, List, Mapping

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from app.agent.confirm import build_confirm_middleware
from app.agent.prompt import system_prompt

logger = logging.getLogger(__name__)

# 工具 → 人性化 status 文案（on_tool_start 用）
TOOL_STATUS: Dict[str, str] = {
    "get_topology": "正在获取集群/租户信息…",
    "get_slow_sql": "正在从 OCP 拉取慢SQL…",
    "get_full_sql_text": "正在从 OCP 拉取完整SQL文本…",
    "get_sql_explain": "正在从 OCP 拉取执行计划…",
    "get_sql_top_plan": "正在从 OCP 拉取SQL计划uid…",
    "get_tenants_list": "正在从 OCP 拉取租户列表…",
    "get_tenant_info": "正在从 OCP 拉取租户信息…",
    "get_clusters_list": "正在从 OCP 拉取集群列表…",

    "execute_sql": "正在执行只读 SQL 查询…",
    "get_table_ddl": "正在获取表结构信息…",
}


def classify_agent_event(event: Mapping) -> dict | None:
    """把一条 astream_events 原始事件翻译为用户事件；不关心的返回 None。"""
    etype = event.get("event")
    if etype == "on_tool_start":
        name = event.get("name", "?")
        text = TOOL_STATUS.get(name, f"正在调用工具 {name}…")
        return {"type": "status", "text": text}
    if etype == "on_chat_model_stream":
        chunk = (event.get("data") or {}).get("chunk")
        if chunk is None:
            return None
        # 正在构造 tool_call 的流式块没有用户文本，跳过
        if getattr(chunk, "tool_call_chunks", None):
            return None
        text = getattr(chunk, "content", "") or ""
        # 真实 AIMessageChunk 的 content 可能是内容块列表（非 str），只透传纯文本增量
        if not isinstance(text, str):
            return None
        if not text:
            return None
        return {"type": "delta", "text": text}
    return None


async def stream_chat(
    model: BaseChatModel,
    tools: List[BaseTool],
    messages: List[BaseMessage],
    *,
    thread_id: str | None = None,
    checkpointer=None,
    max_seconds: int = 120,
    broker=None,
    confirm_enabled: bool = True,
    confirm_timeout_seconds: float = 120,
    recursion_limit: int = 100,
) -> AsyncIterator[dict]:
    """对历史消息运行 agent，产出对外事件流。单轮超时兜底（spec §8.4）。

    thread_id + checkpointer 同时在场时启用会话记忆：本轮 messages 只应包含「新消息」，
    历史由 langgraph 依据 thread_id 从检查点加载并追加（add_messages 语义）；
    二者缺一时按无状态处理，此时 messages 需包含完整历史。

    max_seconds 只约束 agent 执行（生产者），不因消费端停顿/背压而误中止。

    broker 在场且 confirm_enabled 时挂载 DB 操作确认中间件：命中受控工具的调用
    先发 confirm_request 事件并挂起等答复（批准→执行；拒绝/超时→拒绝观察）。
    确认等待等待发生在 agent 侧（生产者任务），不计入消费端背压。

    recursion_limit 以运行时 config 显式注入 astream_events，跨 langgraph 版本稳定，
    避免旧版默认 25 触顶导致 GRAPH_RECURSION_LIMIT 报错。
    """
    from langchain.agents import create_agent

    # 无 profile 信息的模型（如测试桩）不支持 fraction 触发器，退化为绝对阈值
    if getattr(model, "profile", None):
        trig = ("fraction", 0.75)   # 触发压缩：上下文 >= 75% of max_input_tokens
        keep = ("fraction", 0.30)  # 保留：压缩后只保留最近 30% 的上下文
    else:
        trig = ("messages", 24)
        keep = ("messages", 8)
    # 上下文压缩中间件：达到最大输入 75% 时触发摘要，压缩后保留 30% 上下文
    summarization = SummarizationMiddleware(
        model=model,
        trigger=trig,
        keep=keep,
    )
    q: asyncio.Queue = asyncio.Queue()

    async def _produce() -> None:
        # 确认中间件在 producer 内构建：owner = 本 producer task，
        # 其 finally 统一 fail_all——流正常结束/断开/异常都不留挂起确认。
        confirm_mw = build_confirm_middleware(
            emit=q.put_nowait,
            broker=broker if confirm_enabled else None,
            owner=asyncio.current_task(),
            timeout_seconds=confirm_timeout_seconds,
        )
        agent: Runnable = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt(now=datetime.now()),
            middleware=[confirm_mw, summarization],
            checkpointer=checkpointer,
        )
        # 有检查点且给了 thread_id：messages 只含本轮新消息，历史由 langgraph
        # 从该 thread 的状态加载（messages 通道是 add_messages 追加语义）。
        run_config: dict = {"recursion_limit": recursion_limit}
        if checkpointer is not None and thread_id:
            run_config["configurable"] = {"thread_id": thread_id}
        try:
            async with asyncio.timeout(max_seconds):
                # create_agent 返回的是编译后的状态图，输入需按 {"messages": [...]} 传。
                # 显式注入 recursion_limit：langgraph 运行时 config 会覆盖编译默认，
                # 从而规避不同 langgraph 版本的默认差异（旧版默认 25，新版 create_agent 内部写 9999）。
                async for event in agent.astream_events(
                    {"messages": messages},
                    config=run_config,
                    version="v2",
                ):
                    user = classify_agent_event(event)
                    if user is not None:
                        await q.put(user)
                await q.put({"type": "done"})
        except TimeoutError:
            await q.put({"type": "error", "message": f"agent 执行超过 {max_seconds}s，已中止"})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("agent 执行失败: %s", e)
            await q.put({"type": "error", "message": str(e)})
        finally:
            if broker is not None:
                broker.fail_all(asyncio.current_task(), reason="stream_end")

    task = asyncio.create_task(_produce())
    try:
        while True:
            ev = await q.get()
            yield ev
            if ev["type"] in ("done", "error"):
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass