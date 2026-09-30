"""API 后置重排（P2a）：融合后的候选交给重排模型再排一次。

对应设计文档 §8.6。四条约束是设计期定下、实测踩过的：

1. **输入是融合后的前 ``rerank.top_n`` 条**（默认 30），不是各路召回池；重排看的是
   「问题 ↔ 候选段落」的语义相关性，池子太大只是白烧 token。
2. **调用点在融合之后、``MAX_CHUNKS_PER_PATH``/``limit`` 截断之前**：截断前重排才可能
   把「融合排 7、其实最相关」的块提到前面（截断后再排等于白排）。
3. **降级不抛到检索层**：超时/不可达/响应坏掉都只抛 ``RerankUnavailable``，调用方接住后
   **保留融合原序**，只标 ``rerank_degraded``——检索结果宁可是「排得不够好」也不能是 500。
4. **只做一次调用、不重试**：重排站在用户延迟的关键路径上（``timeout_seconds`` 默认 3s），
   像 embedding 那样退避重试会把一次查询拖到十几秒；失败就退融合序。

协议两套（``RerankConfig.protocol``）：
  - ``jina``：POST ``{base_url}/rerank``，``{"model","query","documents","top_n"}``，
    响应 ``results[].index / results[].relevance_score``（Jina / Cohere / 多数兼容端点）。
  - ``dashscope``：POST ``{host}/api/v1/services/rerank/text-rerank/text-rerank``，
    ``{"model","input":{"query","documents"},"parameters":{"top_n",...}}``，
    响应 ``output.results[]...``（阿里云百炼；其 compatible-mode 不提供 /rerank）。
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Protocol, Sequence, runtime_checkable

import requests

from app.config import RerankConfig, load_settings

logger = logging.getLogger(__name__)

#: 单条 passage 的兜底截断宽度（``max_passage_chars`` 配 0 或负数时用）
DEFAULT_MAX_PASSAGE_CHARS = 500

#: 响应体进日志/异常前的截断宽度
_ERROR_TEXT_LIMIT = 240


class RerankUnavailable(RuntimeError):
    """重排不可用（未配置/超时/网络/响应异常）——调用方据此退回融合原序。"""


class RerankResult(tuple):
    """``(index, score)`` 的具名视图，纯粹为了调用方读代码方便。"""

    __slots__ = ()

    def __new__(cls, index: int, score: float) -> "RerankResult":
        return super().__new__(cls, (int(index), float(score)))

    @property
    def index(self) -> int:
        return self[0]

    @property
    def score(self) -> float:
        return self[1]


@runtime_checkable
class Reranker(Protocol):
    """重排客户端协议（测试里塞假实现，检索层不关心底层是哪家 API）。"""

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int = 0
    ) -> list[tuple[int, float]]:
        """返回 ``(documents 下标, 相关性分)``，按分**降序**。

        只包含服务端真正返回的条目；调用方负责把没返回的候选按原序接在后面。
        ``top_n<=0`` 表示不限（由服务端决定）。
        """
        ...

    def close(self) -> None:  # pragma: no cover - 协议声明
        ...


def _truncate(text: str, limit: int = _ERROR_TEXT_LIMIT) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


def _collapse(text: str) -> str:
    """把换行/连续空白压成单空格：passage 里的换行对重排模型只是噪声。"""
    return " ".join(str(text or "").split())


class ApiReranker:
    """HTTP 重排客户端（``requests.Session`` 复用连接，线程安全只需串行调用）。"""

    def __init__(self, config: RerankConfig, *, session: requests.Session | None = None):
        self.config = config
        self.model = config.model
        self.endpoint_url = config.endpoint_url
        self.timeout_seconds = float(config.timeout_seconds)
        self.top_n = max(0, int(config.top_n))
        self.max_passage_chars = int(config.max_passage_chars) or DEFAULT_MAX_PASSAGE_CHARS
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._lock = threading.Lock()
        self._stats = {"requests": 0, "failures": 0, "reranked": 0}

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<ApiReranker model={self.model!r} url={self.endpoint_url!r}>"

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "ApiReranker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 对外接口 ----

    def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int = 0
    ) -> list[tuple[int, float]]:
        if not query or not query.strip():
            raise RerankUnavailable("空查询无法重排")
        if not documents:
            return []
        if not self.model:
            raise RerankUnavailable("rerank 未配置模型（rerank.model）")
        passages = [self._passage(doc) for doc in documents]
        payload = self._payload(query, passages, top_n or self.top_n)
        last_error = ""
        try:
            with self._lock:
                self._stats["requests"] += 1
            response = self._session.post(
                self.endpoint_url, json=payload, headers=self._headers(), timeout=self.timeout_seconds
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}: {_truncate(response.text)}"
            else:
                try:
                    results = self._parse(response.json(), len(documents))
                except RerankUnavailable:
                    with self._lock:
                        self._stats["failures"] += 1
                    raise
                except Exception as exc:  # noqa: BLE001 - 响应体不是预期结构
                    with self._lock:
                        self._stats["failures"] += 1
                    raise RerankUnavailable(f"rerank 响应无法解析：{type(exc).__name__}: {exc}") from exc
                with self._lock:
                    self._stats["reranked"] += len(results)
                return results
        with self._lock:
            self._stats["failures"] += 1
        raise RerankUnavailable(
            f"rerank 请求失败（{self.endpoint_url}）：{last_error}"
        )

    # ---- 内部 ----

    def _passage(self, text: str) -> str:
        body = _collapse(text)
        limit = self.max_passage_chars
        if limit > 0 and len(body) > limit:
            return body[:limit]
        return body

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(self, query: str, documents: list[str], top_n: int) -> dict:
        if self.config.protocol == "dashscope":
            parameters: dict[str, Any] = {"return_documents": False}
            if top_n > 0:
                parameters["top_n"] = int(top_n)
            return {
                "model": self.model,
                "input": {"query": query, "documents": documents},
                "parameters": parameters,
            }
        payload: dict[str, Any] = {"model": self.model, "query": query, "documents": documents}
        if top_n > 0:
            payload["top_n"] = int(top_n)
        return payload

    def _parse(self, data: Any, expected: int) -> list[tuple[int, float]]:
        """两种协议的结果都收成 ``[(index, score)]`` 并对下标做越界保护。"""
        if not isinstance(data, dict):
            raise RerankUnavailable(f"rerank 响应不是对象：{_truncate(str(data))}")
        results = data.get("results")
        if results is None:
            output = data.get("output")
            results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(results, list) or not results:
            raise RerankUnavailable(f"rerank 响应缺少 results：{_truncate(str(data))}")
        parsed: list[tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not 0 <= index < expected:
                logger.debug("rerank 返回越界下标，忽略：%r", item)
                continue
            if score is None:
                continue
            parsed.append(RerankResult(index, float(score)))
        if not parsed:
            raise RerankUnavailable(f"rerank 响应里没有可用条目：{_truncate(str(data))}")
        parsed.sort(key=lambda row: (-row[1], row[0]))
        return parsed


def get_reranker(config: RerankConfig | None = None) -> ApiReranker | None:
    """按配置构造客户端；``off`` 或未配置（缺 base_url/model）返回 None。

    与 ``get_embedding_client`` 同口径：返回 None 不代表错误，调用方据此跳过重排
    （``mode=auto`` 静默等价 off；``mode=api`` 缺配置在启动校验阶段就 fail closed）。
    """
    if config is None:
        try:
            config = load_settings().rerank
        except Exception:  # noqa: BLE001 - 配置读不到不该让检索起不来
            return None
    if not config.enabled:
        return None
    return ApiReranker(config)