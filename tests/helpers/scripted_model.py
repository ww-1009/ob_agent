"""脚本化 Chat 模型：按预设顺序返回消息，供离线端到端测试。

- 预设里 AIMessage 含 tool_calls 时，流式产出对应的 tool_call_chunks，驱动框架 agent 执行工具。
- 预设里普通 AIMessage 时，把 content 拆成小块流式产出，模拟真实流式。
- 不真正调用网络；agent 框架对模型调 bind_tools 时原样返回自身（stub 忽略真实工具绑定，
  答案完全由预设决定，BaseChatModel 默认 bind_tools 会抛 NotImplementedError）。
"""
from __future__ import annotations

import json
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class ScriptedChatModel(BaseChatModel):
    """按预设逐条返回消息。不真正调用网络。"""

    responses: List[BaseMessage]
    _i: int = 0

    def __init__(self, responses: Sequence[BaseMessage]) -> None:
        super().__init__(responses=list(responses))

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        # stub 忽略真实工具绑定：模型行为由预设决定，流式路径不变
        return self

    def _chunk_of(self, msg: BaseMessage) -> AIMessageChunk:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            chunks = []
            for idx, tc in enumerate(msg.tool_calls):
                chunks.append(
                    {
                        "name": tc.get("name", ""),
                        "args": json.dumps(tc.get("args", {}), ensure_ascii=False),
                        "id": tc.get("id") or f"call_{idx}",
                        "index": idx,
                    }
                )
            return AIMessageChunk(content="", tool_call_chunks=chunks)
        return AIMessageChunk(content=msg.content or "")

    def _next_preset(self) -> BaseMessage:
        # 预设耗尽时 loud-fail，避免静默重复最后一条预设导致工具调用死循环
        if self._i >= len(self.responses):
            raise RuntimeError(
                f"ScriptedChatModel 预设耗尽：已调用 {self._i} 次但仅 {len(self.responses)} 条预设"
            )
        msg = self.responses[self._i]
        self._i += 1
        return msg

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self._next_preset()
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        msg = self._next_preset()
        chunk = self._chunk_of(msg)
        if run_manager:
            run_manager.on_llm_new_token(chunk.content or "")
        yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        # 异步路径：每次模型调用产出一个大 chunk（tool_call 或整段文本）。
        msg = self._next_preset()
        yield ChatGenerationChunk(message=self._chunk_of(msg))
        # 末尾空 chunk 用于触发流式 tool_call 的 finalize（若缺可能不执行工具）。
        yield ChatGenerationChunk(message=AIMessageChunk(content=""))
