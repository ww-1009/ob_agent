import pytest
from langchain_openai import ChatOpenAI

from app.agent.model import build_chat_model
from app.config import LLMConfig


def test_missing_config_raises_clear_error():
    with pytest.raises(ValueError, match="llm.*未配置|base_url|api_key"):
        build_chat_model(LLMConfig(base_url="", api_key="", model=""))


def test_returns_chat_openai_instance():
    m = build_chat_model(
        LLMConfig(base_url="https://x/v1", api_key="k", model="m")
    )
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "m"
    assert m.openai_api_base == "https://x/v1"
    assert m.openai_api_key.get_secret_value() == "k"
    assert m.streaming is True
    assert m.max_retries == 1


def test_wires_temperature_from_config():
    m = build_chat_model(LLMConfig(base_url="https://x/v1", api_key="k", model="m", temperature=0.7))
    assert m.temperature == 0.7


@pytest.mark.parametrize(
    "kw",
    [
        {"base_url": "https://x/v1", "api_key": "k"},
        {"base_url": "https://x/v1", "model": "m"},
        {"api_key": "k", "model": "m"},
    ],
)
def test_partial_missing_config_names_the_field(kw):
    cfg = LLMConfig(**kw)
    with pytest.raises(ValueError, match="llm.*未配置|base_url|api_key"):
        build_chat_model(cfg)
