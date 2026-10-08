"""Embedding 客户端：OpenAI 兼容 ``POST {base_url}/embeddings``。

设计取舍：

- **不引入 openai SDK**：只要一个 HTTP POST，requests 已在依赖里；省一个包、也多一点可控性
  （超时、重试、连接复用都自己说了算）。
- **批量上限来自服务端**：百炼实测单请求最多 20 条（``batch size is invalid, it should not be
  larger than 20.``），所以 ``EmbeddingConfig.batch_size`` 默认 16、这里再夹一道 20。
- **失败必须显式**：调用方（索引构建 / 检索）要把「embedding 挂了」降级成「只跑稀疏一路」，
  所以这里统一抛 ``EmbeddingUnavailable``，绝不返回半截向量或零向量填充。
- **查询侧带 LRU 缓存**：同一句话重复问（agent 重试、多轮对话）不必再花 ~200ms；文档侧不进缓存，
  否则 25k 块的向量会把内存吃光。
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests

from app.config import EmbeddingConfig, load_settings

logger = logging.getLogger(__name__)

# 服务端硬上限（百炼兼容模式实测）：单请求 input 最多 20 条。
MAX_BATCH_SIZE = 20
# 失败重试：只对「可能是抖动」的状态码重试，其它 4xx 直接抛（多半是模型名/维度问题）。
# 403 也进重试集：百炼侧「AccessDenied.Unpurchased」实测是**间歇性**的（同一 key 前一刻
# 可用、构建中途突然 403），慢建模式下宁可多等 3 分钟也不要把整次全量构建废掉。
_RETRY_STATUS = frozenset({403, 429, 500, 502, 503, 504})
#: 429 多半是端点 TPM/日配额限速，退避必须够长才可能跨过一个限速窗口（实测并发 4、
#: batch 16 时约 45s 撞 429，旧值 (0.4, 1.2) 三次重试都在同一窗口内，必然全灭）。
#: 服务端给了 ``Retry-After`` 就用它（取二者较大值）。
_RETRY_BACKOFF_SECONDS = (5.0, 20.0, 60.0, 120.0)


class EmbeddingUnavailable(RuntimeError):
    """embedding 不可用（网络、鉴权、额度、服务端 5xx、返回维度不符）。

    调用方捕获它并降级（稀疏一路），不要把它当成 500 抛给用户。
    """


def _truncate(text: str, limit: int = 240) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _retry_after_seconds(response: requests.Response) -> float:
    """解析 ``Retry-After``（只认秒数形式；HTTP-date 形式当没给）。"""
    raw = response.headers.get("Retry-After") or ""
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        return 0.0


def _sleep(seconds: float) -> None:
    """重试等待（单独抽成函数，测试里替换掉真实 sleep）。"""
    threading.Event().wait(seconds)


class ApiEmbeddingClient:
    """线程安全的 embedding 客户端（requests.Session 复用连接）。"""

    def __init__(self, config: EmbeddingConfig, *, session: requests.Session | None = None):
        self.config = config
        self.dims = int(config.dims)
        self.model = config.model
        self.endpoint_url = config.endpoint_url
        self.batch_size = max(1, min(int(config.batch_size), MAX_BATCH_SIZE))
        self.concurrency = max(1, int(config.concurrency))
        self.timeout_seconds = float(config.timeout_seconds)
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._executor: ThreadPoolExecutor | None = None
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_limit = max(0, int(config.query_cache_size))
        self._lock = threading.Lock()
        self._stats = {"requests": 0, "texts": 0, "cache_hits": 0, "retries": 0, "failures": 0}

    # ---- 基本信息 ----

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<ApiEmbeddingClient model={self.model!r} dims={self.dims} url={self.endpoint_url!r}>"

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # ---- 缓存 ----

    def _cache_get(self, text: str) -> list[float] | None:
        if self._cache_limit <= 0:
            return None
        with self._lock:
            hit = self._cache.get(text)
            if hit is None:
                return None
            self._cache.move_to_end(text)
            self._stats["cache_hits"] += 1
            return list(hit)

    def _cache_put(self, text: str, vector: list[float]) -> None:
        if self._cache_limit <= 0:
            return
        with self._lock:
            self._cache[text] = list(vector)
            self._cache.move_to_end(text)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ---- 对外接口 ----

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量（带 LRU 缓存）。"""
        if not text or not text.strip():
            raise EmbeddingUnavailable("空查询无法向量化")
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vector = self._embed_batch([text])[0]
        self._cache_put(text, vector)
        return vector

    def embed_documents(self, texts: list[str], *, progress=None) -> list[list[float]]:
        """批量向量化，返回顺序与入参一致。``progress`` 每完成一批回调 (done, total)。"""
        if not texts:
            return []
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        if len(batches) == 1 or self.concurrency == 1:
            vectors: list[list[float]] = []
            for batch in batches:
                vectors.extend(self._embed_batch(batch))
                if progress is not None:
                    progress(len(vectors), len(texts))
            return vectors

        results: list[list[list[float]] | None] = [None] * len(batches)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._embed_batch, batch): i for i, batch in enumerate(batches)}
            done = 0
            try:
                for future in futures:
                    idx = futures[future]
                    results[idx] = future.result()
                    done += len(batches[idx])
                    if progress is not None:
                        progress(done, len(texts))
            except Exception:
                for future in futures:
                    future.cancel()
                raise
        flat: list[list[float]] = []
        for chunk in results:
            assert chunk is not None
            flat.extend(chunk)
        if len(flat) != len(texts):
            raise EmbeddingUnavailable(f"embedding 返回条数不符：期望 {len(texts)}，实得 {len(flat)}")
        return flat

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "ApiEmbeddingClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- 内部 ----

    def _payload(self, batch: list[str]) -> dict:
        payload: dict = {"model": self.model, "input": batch}
        if self.dims > 0:
            # 有些 OpenAI 兼容服务支持按 dimensions 截断；不支持时会被忽略（服务端忽略未知字段）。
            payload.setdefault("dimensions", self.dims)
        return payload

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        if not self.model:
            raise EmbeddingUnavailable("embedding 未配置模型（embedding.model）")
        payload = self._payload(batch)
        last_error = ""
        retry_after = 0.0
        for attempt in range(len(_RETRY_BACKOFF_SECONDS) + 1):
            try:
                with self._lock:
                    self._stats["requests"] += 1
                response = self._session.post(
                    self.endpoint_url, json=payload, headers=self._headers(), timeout=self.timeout_seconds
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code in _RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}: {_truncate(response.text)}"
                    retry_after = _retry_after_seconds(response)
                elif response.status_code >= 400:
                    with self._lock:
                        self._stats["failures"] += 1
                    raise EmbeddingUnavailable(
                        f"embedding 请求失败 HTTP {response.status_code}（{self.endpoint_url}）：{_truncate(response.text)}"
                    )
                else:
                    try:
                        data = response.json()
                        vectors = self._parse(data, len(batch))
                    except EmbeddingUnavailable:
                        with self._lock:
                            self._stats["failures"] += 1
                        raise
                    except Exception as exc:  # noqa: BLE001 - 响应体不是预期结构
                        with self._lock:
                            self._stats["failures"] += 1
                        raise EmbeddingUnavailable(f"embedding 响应无法解析：{type(exc).__name__}: {exc}") from exc
                    with self._lock:
                        self._stats["texts"] += len(batch)
                    return vectors
            if attempt < len(_RETRY_BACKOFF_SECONDS):
                with self._lock:
                    self._stats["retries"] += 1
                wait = max(_RETRY_BACKOFF_SECONDS[attempt], retry_after)
                retry_after = 0.0
                logger.info(
                    "embedding 第 %d 次重试，等 %.1fs（%s）", attempt + 1, wait, last_error
                )
                _sleep(wait)
        with self._lock:
            self._stats["failures"] += 1
        raise EmbeddingUnavailable(f"embedding 连续 {len(_RETRY_BACKOFF_SECONDS) + 1} 次失败：{last_error}")

    def _parse(self, data: dict, expected: int) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise EmbeddingUnavailable(f"embedding 响应缺少 data：{_truncate(str(data))}")
        ordered = sorted(items, key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0)
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise EmbeddingUnavailable(f"embedding 条目缺少向量：{_truncate(str(item))}")
            values = [float(x) for x in vector]
            if self.dims > 0 and len(values) != self.dims:
                raise EmbeddingUnavailable(f"embedding 维度不符：期望 {self.dims}，实得 {len(values)}")
            vectors.append(values)
        if len(vectors) != expected:
            raise EmbeddingUnavailable(f"embedding 返回条数不符：期望 {expected}，实得 {len(vectors)}")
        return vectors


def get_embedding_client(config: EmbeddingConfig | None = None) -> ApiEmbeddingClient | None:
    """按配置构造客户端；未配置（缺 base_url/model）返回 None，让调用方走稀疏降级。"""
    if config is None:
        try:
            config = load_settings().embedding
        except Exception:  # noqa: BLE001 - 配置缺失不该让检索起不来
            return None
    if not config.is_configured:
        return None
    return ApiEmbeddingClient(config)