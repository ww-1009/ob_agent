"""回归测试：审计发现的三处启动 / 流式缺陷。

- backend/app/config.py：无 config.yaml / .env 时 ocp.provider 缺省必须是 mock
  （与 OcpConfig.provider、README「默认 mock 模式」一致），此前回落 real 会导致
  base_url 为空仍走 real，OCP 工具必然报配置缺失。
- backend/app/agent/runner.py：_produce 内 build_confirm_middleware / create_agent
  的构建异常必须被捕获并投 error 事件、finally 里 fail_all；否则消费端永久阻塞在
  q.get()，max_seconds 完全失效，线程锁被占导致该 thread 永久 409。
- backend/app/memory/__init__.py：min_size > max_size 等非法池参数必须降级为
  「记忆关闭」并返回原因，而不是让 ValueError 穿透 open_memory → lifespan。
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage

from app.agent import runner as runner_mod
from app.agent.runner import stream_chat
from app.config import MemoryConfig, OcpConfig, load_settings
from app.memory import open_memory
from helpers.scripted_model import ScriptedChatModel


def test_missing_config_defaults_ocp_provider_to_mock(tmp_path):
    # config.yaml 不存在且无环境变量覆盖：应回落 mock，而非 real
    missing = tmp_path / "config.yaml"
    s = load_settings(config_path=str(missing), env={})
    assert s.ocp.provider == "mock"
    assert s.ocp.provider == OcpConfig().provider
    assert s.sql_ro.provider == "mock"


class _RecordingBroker:
    """只记录 fail_all 调用，不关心 Future 细节。"""

    def __init__(self) -> None:
        self.fail_all_reasons: list[str] = []

    def fail_all(self, owner=None, *, reason: str = "stream_closed") -> list:
        self.fail_all_reasons.append(reason)
        return []


@pytest.mark.asyncio
async def test_producer_construction_failure_yields_error_not_hang(monkeypatch):
    # 修复前：异常在 try 之外抛出 → 无 error 事件、无 fail_all，消费端死等。
    def boom(**kwargs):
        raise RuntimeError("middleware construction failed")

    monkeypatch.setattr(runner_mod, "build_confirm_middleware", boom)
    broker = _RecordingBroker()
    model = ScriptedChatModel(responses=[AIMessage(content="never reached")])
    events = []
    async with asyncio.timeout(5):  # 修复前这里会触发 TimeoutError
        async for ev in stream_chat(model, [], [], broker=broker):
            events.append(ev)
    assert events and events[-1]["type"] == "error"
    assert broker.fail_all_reasons == ["stream_end"]


@pytest.mark.asyncio
async def test_invalid_pool_bounds_degrade_instead_of_crashing():
    cfg = MemoryConfig(
        enabled=True,
        dbname="x",
        host="127.0.0.1",
        pool_min_size=3,
        pool_max_size=1,
    )
    # 修复前：AsyncConnectionPool 构造抛 ValueError，直接穿透 open_memory
    runtime, error = await open_memory(cfg)
    assert runtime is None
    assert error and "pool_min_size" in error