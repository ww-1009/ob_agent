"""Embedding 客户端（app.agent.embedding）契约测试。

全部用假 Session 跑，不碰真实端点：这里固化的是**协议细节与失败语义**——
端点拼接、payload/鉴权头、index 排序、维度与条数校验、批量上限、重试策略、
查询 LRU 缓存、以及「失败一律抛 EmbeddingUnavailable 让上层降级」。
"""
from __future__ import annotations

import json
import threading
import time

import pytest
import requests

from app.agent import embedding as embedding_mod
from app.agent.embedding import (
    MAX_BATCH_SIZE,
    ApiEmbeddingClient,
    EmbeddingUnavailable,
    get_embedding_client,
)
from app.config import EmbeddingConfig


def cfg(**over) -> EmbeddingConfig:
    base = dict(
        base_url="https://api.example.com/compatible-mode/v1/",
        api_key="secret",
        model="emb-test",
        dims=3,
        batch_size=4,
        concurrency=1,
        timeout_seconds=5,
        query_cache_size=2,
    )
    base.update(over)
    return EmbeddingConfig(**base)


class FakeResponse:
    def __init__(self, status: int = 200, payload: dict | None = None, text: str = "", headers: dict | None = None):
        self.status_code = status
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = dict(headers or {})

    def json(self):
        if self._payload is None:
            raise ValueError("响应体不是 JSON")
        return self._payload


def body(vectors: list[list[float]], *, reverse: bool = False) -> dict:
    items = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if reverse:
        items.reverse()
    return {"data": items}


class FakeSession:
    """按脚本依次返回响应；线程安全，记录每次 POST 的入参。"""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.closed = False
        self._lock = threading.Lock()

    def post(self, url, json=None, headers=None, timeout=None):
        with self._lock:
            self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            item = self.responses.pop(0) if self.responses else FakeResponse(500, text="script exhausted")
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class RuleSession:
    """按请求里的 input 现场生成向量（首元素 = 该文本首字符的码点），可注入失败与乱序延迟。"""

    def __init__(self, *, dims: int = 3, fail_first: int = 0, delay_by_prefix: dict[str, float] | None = None):
        self.dims = dims
        self.fail_first = fail_first
        self.delay_by_prefix = dict(delay_by_prefix or {})
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    def post(self, url, json=None, headers=None, timeout=None):
        with self._lock:
            idx = len(self.calls)
            self.calls.append(json)
            failing = idx < self.fail_first
        if failing:
            return FakeResponse(503, text="busy")
        # 延迟按 batch 首文本的首字符决定（并发下调用顺序不稳定，不能按下标）
        delay = self.delay_by_prefix.get(str(json["input"][0])[:1], 0.0)
        if delay:
            time.sleep(delay)
        data = [
            {"index": i, "embedding": [float(ord(t[0])), 1.0, 2.0][: self.dims]}
            for i, t in enumerate(json["input"])
        ]
        return FakeResponse(200, {"data": data})

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """重试策略照测，但别真的等 1.6 秒。"""
    monkeypatch.setattr(embedding_mod, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))


# ---------------------------------------------------------------- 协议


def test_batch_size_clamped_and_endpoint_joined():
    client = ApiEmbeddingClient(cfg(batch_size=64), session=FakeSession([]))
    assert client.batch_size == MAX_BATCH_SIZE
    # base_url 末尾的斜杠不能拼出 //embeddings
    assert client.endpoint_url == "https://api.example.com/compatible-mode/v1/embeddings"


def test_payload_and_auth_header():
    session = FakeSession([FakeResponse(200, body([[1.0, 2.0, 3.0]]))])
    client = ApiEmbeddingClient(cfg(), session=session)
    assert client.embed_query("事务隔离") == [1.0, 2.0, 3.0]
    call = session.calls[0]
    assert call["url"] == "https://api.example.com/compatible-mode/v1/embeddings"
    assert call["json"] == {"model": "emb-test", "input": ["事务隔离"], "dimensions": 3}
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["timeout"] == 5


def test_empty_api_key_omits_authorization_header():
    session = FakeSession([FakeResponse(200, body([[1.0, 2.0, 3.0]]))])
    client = ApiEmbeddingClient(cfg(api_key=""), session=session)
    client.embed_query("x")
    assert "Authorization" not in session.calls[0]["headers"]


def test_parse_orders_by_index():
    session = FakeSession([FakeResponse(200, body([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], reverse=True))])
    client = ApiEmbeddingClient(cfg(), session=session)
    assert client._embed_batch(["a", "b"]) == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


def test_dimension_mismatch_raises():
    session = FakeSession([FakeResponse(200, body([[1.0, 2.0]]))])
    client = ApiEmbeddingClient(cfg(), session=session)
    with pytest.raises(EmbeddingUnavailable, match="维度不符"):
        client.embed_query("x")


def test_missing_data_raises():
    session = FakeSession([FakeResponse(200, {"unexpected": True})])
    client = ApiEmbeddingClient(cfg(), session=session)
    with pytest.raises(EmbeddingUnavailable, match="缺少 data"):
        client.embed_query("x")


# ---------------------------------------------------------------- 缓存与批量


def test_query_cache_hit_and_lru_eviction():
    session = FakeSession([FakeResponse(200, body([[float(i)] * 3])) for i in range(4)])
    client = ApiEmbeddingClient(cfg(query_cache_size=2), session=session)
    client.embed_query("q1")
    client.embed_query("q1")  # 命中缓存，不再发请求
    assert client.stats()["requests"] == 1
    assert client.stats()["cache_hits"] == 1
    client.embed_query("q2")
    client.embed_query("q3")  # LRU 上限 2 → q1 被挤出
    client.embed_query("q1")  # 重新请求
    assert client.stats()["requests"] == 4


def test_cache_disabled_when_size_zero():
    session = FakeSession([FakeResponse(200, body([[1.0, 1.0, 1.0]])) for _ in range(2)])
    client = ApiEmbeddingClient(cfg(query_cache_size=0), session=session)
    client.embed_query("q")
    client.embed_query("q")
    assert client.stats()["requests"] == 2
    assert client.stats()["cache_hits"] == 0


def test_embed_documents_batching_order_and_progress():
    session = RuleSession()
    client = ApiEmbeddingClient(cfg(batch_size=2, concurrency=1), session=session)
    seen: list[tuple[int, int]] = []
    vectors = client.embed_documents(["a1", "a2", "b1", "b2", "c1"], progress=lambda d, t: seen.append((d, t)))
    assert [v[0] for v in vectors] == [float(ord(c)) for c in "aabbc"]
    assert [call["input"] for call in session.calls] == [["a1", "a2"], ["b1", "b2"], ["c1"]]
    assert seen == [(2, 5), (4, 5), (5, 5)]
    assert client.stats() == {"requests": 3, "texts": 5, "cache_hits": 0, "retries": 0, "failures": 0}


def test_embed_documents_concurrent_keeps_input_order():
    # a 批最慢、b 批最快：完成顺序被故意打乱，结果仍须按入参顺序返回
    session = RuleSession(delay_by_prefix={"a": 0.08, "b": 0.0, "c": 0.03})
    client = ApiEmbeddingClient(cfg(batch_size=2, concurrency=3), session=session)
    vectors = client.embed_documents(["a1", "a2", "b1", "b2", "c1", "c2"])
    assert [v[0] for v in vectors] == [float(ord(c)) for c in "aabbcc"]


def test_embed_documents_empty_returns_empty():
    client = ApiEmbeddingClient(cfg(), session=FakeSession([]))
    assert client.embed_documents([]) == []


# ---------------------------------------------------------------- 失败语义


def test_retry_then_success():
    session = RuleSession(fail_first=1)
    client = ApiEmbeddingClient(cfg(), session=session)
    assert client.embed_query("x") == [float(ord("x")), 1.0, 2.0]
    assert client.stats()["retries"] == 1
    assert client.stats()["requests"] == 2


def test_retry_exhausted_raises():
    session = FakeSession([FakeResponse(503, text="busy")] * 3)
    client = ApiEmbeddingClient(cfg(), session=session)
    with pytest.raises(EmbeddingUnavailable, match="连续 3 次失败"):
        client.embed_query("x")
    assert len(session.calls) == 3
    assert client.stats()["failures"] == 1


def test_rate_limit_retry_honors_retry_after(monkeypatch):
    """429 的 ``Retry-After`` 必须盖过默认退避：端点 TPM 限速窗口比默认退避长得多。"""
    waits: list[float] = []
    monkeypatch.setattr(embedding_mod, "_sleep", waits.append)
    session = FakeSession(
        [
            FakeResponse(429, text="Allocated quota exceeded", headers={"Retry-After": "42"}),
            FakeResponse(200, payload=body([[1.0, 1.0, 1.0]])),
        ]
    )
    client = ApiEmbeddingClient(cfg(), session=session)
    assert client.embed_query("x") == [1.0, 1.0, 1.0]
    assert waits == [42.0]  # 42 > 默认退避第一档，取大的
    # HTTP-date 形式的 Retry-After 认不出来时退回默认退避（不能当作 0 秒立即重试）
    assert embedding_mod._retry_after_seconds(
        FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
    ) == 0.0
    assert embedding_mod._retry_after_seconds(FakeResponse(429)) == 0.0


def test_client_error_not_retried():
    session = FakeSession([FakeResponse(400, text="model not found")])
    client = ApiEmbeddingClient(cfg(), session=session)
    with pytest.raises(EmbeddingUnavailable, match="HTTP 400"):
        client.embed_query("x")
    assert len(session.calls) == 1


def test_network_error_retried_then_raises():
    session = FakeSession([requests.ConnectionError("boom")] * 3)
    client = ApiEmbeddingClient(cfg(), session=session)
    with pytest.raises(EmbeddingUnavailable, match="连续 3 次失败"):
        client.embed_query("x")
    assert len(session.calls) == 3


def test_empty_query_rejected():
    client = ApiEmbeddingClient(cfg(), session=FakeSession([]))
    with pytest.raises(EmbeddingUnavailable, match="空查询"):
        client.embed_query("   ")


def test_missing_model_raises_without_http():
    session = FakeSession([])
    client = ApiEmbeddingClient(cfg(model=""), session=session)
    with pytest.raises(EmbeddingUnavailable, match="未配置模型"):
        client.embed_query("x")
    assert session.calls == []


def test_get_embedding_client_requires_configuration():
    assert get_embedding_client(EmbeddingConfig()) is None
    assert get_embedding_client(EmbeddingConfig(base_url="", model="m")) is None
    # api_key 故意可选：内网网关常常不需要
    client = get_embedding_client(EmbeddingConfig(base_url="https://x/v1", model="m", dims=3))
    assert isinstance(client, ApiEmbeddingClient)
    client.close()


def test_close_only_closes_owned_session():
    injected = FakeSession([])
    client = ApiEmbeddingClient(cfg(), session=injected)
    client.close()
    assert injected.closed is False