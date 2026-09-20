import json

from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from langchain_core.messages import AIMessage
from helpers.scripted_model import ScriptedChatModel


# 指向一个必然不存在的配置文件，避免读取开发者本机 backend/config.yaml（含真实 LLM 配置会破坏 503 测试）
_MISSING_CONFIG = "/nonexistent-ob-agent-test-config.yaml"


def _app():
    settings = load_settings(config_path=_MISSING_CONFIG, env={"OCP_PROVIDER": "mock", "SQL_PROVIDER": "mock"})
    settings.agent.max_seconds = 5
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "query_slow_sql", "args": {"top_n": 2, "tenant_id": "1001"}, "id": "c1",
                }],
            ),
            AIMessage(content="## 诊断结果\n发现慢SQL sq-scan-orders-1。"),
        ]
    )
    return create_app(settings, model=model)


def test_health():
    client = TestClient(_app())
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ocp_provider"] == "mock"


def test_chat_without_llm_returns_503():
    # 未注入模型且 llm 未配置 → 启动可用但聊天明确报 503（而不是 import 崩溃）
    settings = load_settings(config_path=_MISSING_CONFIG, env={})
    client = TestClient(create_app(settings))
    r = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_chat_streams_to_done():
    client = TestClient(_app())
    body = {"messages": [{"role": "user", "content": "有哪些慢SQL？"}]}
    with client.stream("POST", "/api/chat", json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in resp.iter_lines() if ln.startswith("data: ")]
    events = [json.loads(ln[6:]) for ln in lines]
    types = [e["type"] for e in events]
    assert "done" in types
    assert "error" not in types
    delta = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    assert "sq-scan-orders-1" in delta


def test_unsupported_role_rejected():
    client = TestClient(_app())
    body = {"messages": [{"role": "system", "content": "x"}]}
    r = client.post("/api/chat", json=body)
    assert r.status_code == 422
