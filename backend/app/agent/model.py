"""按配置构建 OpenAI 兼容的 Chat 模型（供国产模型 API 使用）。"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import LLMConfig


def build_chat_model(config: LLMConfig) -> BaseChatModel:
    # YAML 引号值不 trim（env 路径已过滤空白），这里是程序化构造与引号 YAML 的最后防线
    parts = {n: getattr(config, n).strip() for n in ("base_url", "api_key", "model")}
    missing = [k for k, v in parts.items() if not v]
    if missing:
        raise ValueError(f"llm 未配置完整（缺: {', '.join(missing)}）。请在 config.yaml / .env 填写")
    # 瞬时错误自动重试 1 次（spec §8.2）；streaming 供 SSE 流式增量推送。
    return ChatOpenAI(
        model=parts["model"],
        base_url=parts["base_url"],
        api_key=parts["api_key"],
        temperature=config.temperature,
        max_retries=1,
        streaming=True,
        profile={
            "max_input_tokens": config.max_input_tokens,
        },
    )