"""Rerank 客户端（app.agent.rerank）契约测试。

全部用假 Session 跑，不碰真实端点。这里固化的是**协议细节与失败语义**：
两套协议（jina / dashscope）的端点与 payload、鉴权头、结果排序、下标越界保护、
passage 截断，以及「失败一律抛 RerankUnavailable 让检索层退回融合原序」。
"""
from __future__ import annotations

import json

import pytest
import requests

from app.agent.rerank import (
    DEFAULT_MAX_PASSAGE_CHARS,
    ApiReranker,
    RerankUnavailable,
    Reranker,
    get_reranker,
)
from app.config import RerankConfig


def cfg(**over) -> RerankConfig:
    base = dict(
        mode="api",
        protocol="jina",
        base_url="https://api.example.com/v1/",
        api_key="secret",
        model="rerank-test",
        top_n=3,
        timeout_seconds=2.0,
        max_passage_chars=20,
    )
    base.update(over)
    return RerankConfig(**base)


class FakeResponse:
    def __init__(self, status: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("响应体不是 JSON")
        return self._payload


class FakeSession:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.closed = False

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.responses.pop(0) if self.responses else FakeResponse(500, text="script exhausted")
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


def client(responses: list, **over) -> tuple[ApiReranker, FakeSession]:
    session = FakeSession(responses)
    return ApiReranker(cfg(**over), session=session), session


# ---------------------------------------------------------------- 构造与配置


def test_get_reranker_returns_none_when_off_or_unconfigured():
    assert get_reranker(cfg(mode="off")) is None
    assert get_reranker(cfg(base_url="")) is None
    assert get_reranker(cfg(model="")) is None


def test_get_reranker_returns_client_when_configured():
    reranker = get_reranker(cfg())
    assert isinstance(reranker, Reranker)
    assert isinstance(reranker, ApiReranker)
    assert reranker.model == "rerank-test"


def test_auto_mode_with_config_creates_client():
    # auto = 配置齐全即启用，与 api 的差别只在启动校验（fail closed），运行期同一条路
    assert isinstance(get_reranker(cfg(mode="auto")), ApiReranker)


# ---------------------------------------------------------------- jina 协议


def test_jina_payload_endpoint_and_parsing():
    reranker, session = client([FakeResponse(200, {"results": [
        {"index": 2, "relevance_score": 0.2},
        {"index": 0, "relevance_score": 0.9},
    ]})])
    out = reranker.rerank("怎么调优", ["甲", "乙", "丙"])

    call = session.calls[0]
    assert call["url"] == "https://api.example.com/v1/rerank"
    assert call["json"] == {
        "model": "rerank-test",
        "query": "怎么调优",
        "documents": ["甲", "乙", "丙"],
        "top_n": 3,
    }
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["timeout"] == 2.0
    # 服务端顺序不可信，客户端按分降序
    assert out == [(0, 0.9), (2, 0.2)]


def test_explicit_top_n_overrides_config():
    reranker, session = client([FakeResponse(200, {"results": [{"index": 0, "relevance_score": 1.0}]})])
    reranker.rerank("q", ["甲", "乙"], top_n=1)
    assert session.calls[0]["json"]["top_n"] == 1


def test_rerank_without_api_key_omits_authorization():
    reranker, session = client(
        [FakeResponse(200, {"results": [{"index": 0, "relevance_score": 1.0}]})], api_key=""
    )
    reranker.rerank("q", ["甲"])
    assert "Authorization" not in session.calls[0]["headers"]


# ---------------------------------------------------------------- dashscope 协议


def test_dashscope_endpoint_derived_from_compatible_mode_base():
    reranker, session = client(
        [FakeResponse(200, {"output": {"results": [{"index": 1, "relevance_score": 0.7}]}})],
        protocol="dashscope",
        base_url="https://ws-abc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    out = reranker.rerank("q", ["甲", "乙"])

    assert session.calls[0]["url"] == (
        "https://ws-abc.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    assert session.calls[0]["json"] == {
        "model": "rerank-test",
        "input": {"query": "q", "documents": ["甲", "乙"]},
        "parameters": {"return_documents": False, "top_n": 3},
    }
    assert out == [(1, 0.7)]


def test_dashscope_endpoint_kept_when_already_native():
    reranker, _ = client([], protocol="dashscope", base_url="https://host/api/v1/services/rerank/x")
    assert reranker.endpoint_url == "https://host/api/v1/services/rerank/x"


def test_score_field_fallback_accepted():
    # 兼容端点偶尔只回 score（Cohere 老版本）；缺分或下标越界的条目直接丢掉
    reranker, _ = client([FakeResponse(200, {"results": [
        {"index": 0, "score": 0.5},
        {"index": 9, "relevance_score": 0.99},
        {"index": 1},
    ]})])
    assert reranker.rerank("q", ["甲", "乙"]) == [(0, 0.5)]


# ---------------------------------------------------------------- passage


def test_passage_is_collapsed_and_truncated():
    reranker, session = client([FakeResponse(200, {"results": [{"index": 0, "relevance_score": 1.0}]})])
    reranker.rerank("q", ["标题   正文\n第二行很长很长很长很长很长"])
    assert session.calls[0]["json"]["documents"] == ["标题 正文 第二行很长很长很长很长很长"[:20]]


def test_max_passage_chars_zero_falls_back_to_default():
    reranker, _ = client([], max_passage_chars=0)
    assert reranker.max_passage_chars == DEFAULT_MAX_PASSAGE_CHARS


# ---------------------------------------------------------------- 失败语义


def test_request_exception_becomes_rerank_unavailable():
    reranker, _ = client([requests.ConnectionError("boom")])
    with pytest.raises(RerankUnavailable) as excinfo:
        reranker.rerank("q", ["甲"])
    assert "ConnectionError" in str(excinfo.value)


def test_http_error_becomes_rerank_unavailable_with_body():
    reranker, _ = client([FakeResponse(429, text='{"error":"rate limited"}')])
    with pytest.raises(RerankUnavailable) as excinfo:
        reranker.rerank("q", ["甲"])
    assert "HTTP 429" in str(excinfo.value)
    assert "rate limited" in str(excinfo.value)


def test_non_json_response_becomes_rerank_unavailable():
    reranker, _ = client([FakeResponse(200, text="<html>502</html>")])
    with pytest.raises(RerankUnavailable):
        reranker.rerank("q", ["甲"])


def test_missing_results_becomes_rerank_unavailable():
    reranker, _ = client([FakeResponse(200, {"ok": True})])
    with pytest.raises(RerankUnavailable):
        reranker.rerank("q", ["甲"])


def test_all_indices_out_of_range_becomes_rerank_unavailable():
    reranker, _ = client([FakeResponse(200, {"results": [{"index": 5, "relevance_score": 1.0}]})])
    with pytest.raises(RerankUnavailable):
        reranker.rerank("q", ["甲"])


def test_empty_documents_skips_the_call():
    reranker, session = client([])
    assert reranker.rerank("q", []) == []
    assert session.calls == []


def test_empty_query_is_rejected():
    reranker, _ = client([])
    with pytest.raises(RerankUnavailable):
        reranker.rerank("   ", ["甲"])


def test_unconfigured_model_is_rejected_before_http():
    reranker, session = client([], model="")
    with pytest.raises(RerankUnavailable):
        reranker.rerank("q", ["甲"])
    assert session.calls == []


def test_close_only_closes_own_session():
    reranker, session = client([])
    reranker.close()
    assert session.closed is False  # 外部注入的 session 由调用方负责

    own = ApiReranker(cfg())
    own.close()  # 自建 session 关闭不抛异常即可

    with ApiReranker(cfg()) as ctx:
        assert isinstance(ctx, ApiReranker)


def test_stats_counts_requests_failures_reranked():
    reranker, _ = client([
        FakeResponse(200, {"results": [{"index": 0, "relevance_score": 1.0}]}),
        FakeResponse(500, text="nope"),
    ])
    reranker.rerank("q", ["甲"])
    with pytest.raises(RerankUnavailable):
        reranker.rerank("q", ["甲"])
    assert reranker.stats() == {"requests": 2, "failures": 1, "reranked": 1}