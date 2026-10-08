"""Milvus 检索层：稀疏（BM25）与稠密（向量）两路召回 → 自定义 RRF 融合 → 后置调整。

对应设计文档 ``backend/eval/docs/milvus-only-retrieval-design.md`` §8.1 / §8.2。

四条约束是实测踩出来的，别随手改：

1. **稠密一路恒 ``kind == "doc"``**：导航行（``index.md`` / ``README.md``）按契约写进
   ``ob_chunks`` 但向量恒为零向量（FLOAT_VECTOR 不可空），零向量参与 cosine 排序没有意义，
   导航页只从稀疏一路召回；``include_index=True`` 时靠稀疏一路把它们捞回来。
2. **后置调整必须在融合之前、且在各路内归一化之后**：RRF 只看名次，而两路距离量纲差一个
   数量级（真库实测 BM25 ≈13、cosine ≈0.58），直接往原始距离上加 ``version_match_bonus``
   这类常量，会让稠密一路的版本命中行无差别置顶。这里先把每路 pool 的距离线性归一化到
   ``[0, 1]``，再按配置分值除以 ``SCORE_SPAN_REFERENCE``（FTS5 侧 ``-bm25`` 的典型跨度）
   施加，两路才可比。
3. **降级不抛异常**：embedding 不可用 → 只跑稀疏；Milvus 打不开/索引缺失 → 空结果 +
   ``degraded`` 标记（工具层照常 ok，绝不 500）。
4. **自建 RRF 而不是内建 RRFRanker**：内建 ranker 只接受 pymilvus 的 ``AnnSearchRequest``，
   没法把后置调整插进「各路排序之前」；M3 实测内建 ``RRFRanker`` 与自建 RRF 的 top10 完全
   一致，所以自建没有质量代价。
5. **列权重重排只在稀疏一路**（``rank_route(column_tokens=...)``）：Milvus 的 BM25 只有
   ``text`` 一个字段，没有 FTS5 那种 ``bm25(title 10, keywords 6, section 4, body 1)`` 的
   列权重，靠重复标题也补不回来；检索期用 title/keywords/section/body 四列做一次加权命中
   并与归一化距离混合，179 条基线 sparse 一路从 70.95% / 0.607 提到 88.27% / 0.768。
   稠密一路不打列分（保持纯语义排序），要打也必须在两路各自的 rank_route 里打。

条目形状（``path``/``kind``/``section``/``title``/``mode``/``version``/``score``/``snippet``）
与 FTS5 路径保持一致；``score`` 是 RRF 融合分（越大越好），与 FTS5 的 ``-bm25 + 奖励``
**不同量纲、不可跨引擎比数值**。``sources`` / ``sparse_rank`` / ``dense_rank`` 是给 M8
评测报告用的诊断字段。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from app.agent.embedding import get_embedding_client
from app.agent.milvus_index import (
    FIELD_KIND,
    FIELD_KEYWORDS,
    FIELD_MODE,
    FIELD_PATH,
    FIELD_PK,
    FIELD_SECTION,
    FIELD_TEXT,
    FIELD_TITLE,
    FIELD_VECTOR,
    FIELD_VERSION,
    FIELD_SPARSE,
    MilvusIndex,
    MilvusIndexMissing,
    MilvusUnavailable,
    get_milvus_index,
)
from app.agent.rerank import get_reranker
from app.config import RetrievalConfig, Settings, load_settings

logger = logging.getLogger(__name__)

#: 支持的检索器；M7 已删掉 FTS5，这里是唯一实现
RETRIEVERS = ("sparse", "dense", "hybrid")

#: 重排模式（与 ``RerankConfig.mode`` 同名单）：``""`` 取配置，``auto`` 配置齐全才启用，
#: ``off`` 强制关闭，``api`` 强制启用（未配置时只标降级、不抛）
RERANK_MODES = ("auto", "off", "api")

#: 后置调整的归一化基准：FTS5 侧 ``-bm25`` 的典型跨度（179 条基线实测）
SCORE_SPAN_REFERENCE = 20.0

#: 列权重重排（M6 实测的核心质量来源）。FTS5 的 bm25() 用四列权重
#: ``(title 10, keywords 6, section 4, body 1)``；Milvus 的 BM25 只有一个 ``text`` 字段，
#: 靠重复标题（``TITLE_REPEAT``）近似列权重，仍然丢掉了「命中在哪一列」——179 条基线实测
#: sparse 一路 hit@5 70.95%、MRR 0.607（FTS5 79.33% / 0.706）。补救办法是拿库里的
#: title/keywords/section/body 四列在**客户端**算一次加权命中，与归一化距离线性混合：
#: 加这一项后 sparse 一路 88.27% / 0.768，全面超过 FTS5（含 MRR）。
#: 权重与 FTS5 完全一致，别随手调；``COLUMN_OVERLAP_ALPHA`` 实测 0.5~0.6 是平台（0.3 → 84.4%，
#: 0.4 → 87.7%，0.5/0.6 → 87.7%，纯重叠 83.8%），取中值。
COLUMN_WEIGHTS: dict[str, float] = {
    "title": 10.0,
    "keywords": 6.0,
    "section": 4.0,
    "body": 1.0,
}
#: 列权重和：重叠分归一化到 [0,1] 的基准
COLUMN_SCORE_SPAN = sum(COLUMN_WEIGHTS.values())
#: 重叠分与「归一化距离 + 后置调整」的混合比例（0 = 只用距离，1 = 只用重叠）
COLUMN_OVERLAP_ALPHA = 0.5

#: 摘要宽度，与 FTS5 路径的 ``_excerpt`` 默认值一致
SNIPPET_WIDTH = 220

#: 从 Milvus 取回的输出字段（正文用它做摘要，其余原样透传给条目）
OUTPUT_FIELDS = (
    FIELD_PK,
    FIELD_PATH,
    FIELD_SECTION,
    FIELD_TITLE,
    FIELD_KEYWORDS,
    FIELD_KIND,
    FIELD_MODE,
    FIELD_VERSION,
    FIELD_TEXT,
)

#: Milvus 打不开/索引缺失——都当作可降级错误
_MILVUS_ERRORS = (MilvusUnavailable, MilvusIndexMissing)

#: jieba 模块（首次 ``_column_tokens`` 时加载）
_JIEBA: Any = None

#: ``reranker`` 参数缺省哨兵：区分「没传」（按配置懒构造）与「显式 None」（测试里不重排）
_RERANKER_UNSET: Any = object()


def _docs():
    """延迟 import doc_index 的几个纯工具（doc_index.search 会反向 import 本模块）。"""
    from app.agent import doc_index

    return doc_index


def expand_query(query: str) -> str:
    """把命中的同义词展开词追加到查询串后面（FTS5 的「同义词另一路」在 Milvus 里做查询扩展）。

    只追加、不改写原词：原词留在最前面，BM25 的短语邻近度仍偏向原词命中。
    """
    doc_index = _docs()
    aliases: list[str] = []
    seen: set[str] = set()
    for key, values in doc_index._SYNONYMS.items():
        if key not in query:
            continue
        for alias in values:
            lowered = alias.lower()
            if lowered in seen or lowered in query.lower():
                continue
            seen.add(lowered)
            aliases.append(alias)
    if not aliases:
        return query
    return f"{query} {' '.join(aliases)}"


def _column_tokens(query: str) -> list[str]:
    """列权重重排用的查询词：jieba 分词，丢掉纯标点与单个英文字母。

    为什么是 jieba：Milvus 的 ``text`` 字段用的就是 jieba 分析器，客户端用同一套分词，
    「查询切出来的词」与「语料索引里的词」才会对齐；FTS5 时代的 ``_query_tokens``（单字 +
    双字，M7 已删）在这里反而不合用——实测拿它做重叠分，hit@5 只有 68.7%，literal/mode/
    version 全面下跌。
    jieba 首次加载约 0.5s，放在懒加载里（模块 import 时不付这个钱）。
    """
    if not query or not query.strip():
        return []
    global _JIEBA
    try:
        if _JIEBA is None:
            import jieba  # noqa: PLC0415 - 首次加载约 0.5s，不能拖慢启动

            _JIEBA = jieba
        words = _JIEBA.lcut(query)
    except Exception as exc:  # noqa: BLE001 - 分词挂了不该让检索失败
        logger.warning("jieba 分词失败，跳过列权重重排：%s", exc)
        return []
    tokens: list[str] = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in word):
            continue
        if word.isascii() and len(word) < 2:
            continue
        tokens.append(word)
    return tokens


def column_overlap_score(hit: "Hit", tokens: Sequence[str]) -> float:
    """一条命中在查询词上的列加权覆盖率（归一化到 [0, 1]，越大越好）。

    一个词可以同时命中多列（标题 + 正文是常见组合），与 FTS5 的 bm25 求和同向；
    keywords 用 ``elif``——它只是标题的补充信号，重复计分会把关键词堆出来的文档抬过头。
    """
    if not tokens:
        return 0.0
    total = 0.0
    for token in tokens:
        if token in hit.title:
            total += COLUMN_WEIGHTS["title"]
        elif token in hit.keywords:
            total += COLUMN_WEIGHTS["keywords"]
        if token in hit.section:
            total += COLUMN_WEIGHTS["section"]
        if token in hit.body:
            total += COLUMN_WEIGHTS["body"]
    return total / COLUMN_SCORE_SPAN


def _rerank_passage(hit: "Hit", max_chars: int) -> str:
    """送进重排模型的单条 passage：``标题 > 小节`` 起头，后接正文（压空白、按上限截断）。

    标题在前是有意的：重排模型对开头更敏感，而「标题 > 小节」正是 FTS5 时代列权重的语义
    浓缩（正文里同样的标题被重复了三遍，但那只是 BM25 的活儿）。
    """
    head = " > ".join(part.strip() for part in (hit.title, hit.section) if part and part.strip())
    body = " ".join((hit.body or "").split())
    text = f"{head}\n{body}" if head else body
    limit = int(max_chars or 0)
    if limit > 0 and len(text) > limit:
        return text[:limit]
    return text


def _escape(value: str) -> str:
    """Milvus 表达式里的字符串字面量：双引号与反斜杠会破坏表达式。"""
    return str(value).replace("\\", "").replace('"', "")


def _scalar_filter(*, mode: str = "", version: str = "", include_index: bool = False) -> str:
    """标量过滤表达式。

    ``include_index=False`` 时导航页**整行排除**（不是降权）：导航页抢 top1 是硬门禁，
    而 RRF 只看名次、pool 很小时被降权的导航行仍可能进来，只有排除才是保证。需要清单类
    答案的提问由调用方传 ``include_index=True``。
    """
    kind = 'kind in ["doc","nav"]' if include_index else 'kind == "doc"'
    parts = [kind]
    if mode:
        parts.append(f'(mode == "{_escape(_docs()._normalize_mode(mode))}" or mode == "")')
    if version:
        normalized = str(version).strip().lstrip("Vv")
        parts.append(f'(version == "" or version like "{_escape(normalized)}%")')
    return " and ".join(parts)


# ---------------------------------------------------------------- 数据类型


@dataclass
class Hit:
    """一条召回行（两路共用一个结构，距离只在所属路内有意义）。"""

    pk: int
    path: str
    kind: str
    section: str
    title: str
    mode: str
    version: str
    body: str
    keywords: str = ""
    distance: float = 0.0

    @classmethod
    def from_entity(cls, hit: Mapping[str, Any]) -> Hit:
        entity = hit.get("entity") or {}
        text = str(entity.get(FIELD_TEXT) or "")
        # text = 加权前缀 + " | " + 正文；摘要只看正文，别把重复三遍的标题塞进 snippet
        body = text.split(" | ", 1)[1] if " | " in text else text
        return cls(
            pk=int(hit.get(FIELD_PK) if hit.get(FIELD_PK) is not None else entity.get(FIELD_PK, 0)),
            path=str(entity.get(FIELD_PATH) or ""),
            kind=str(entity.get(FIELD_KIND) or "doc"),
            section=str(entity.get(FIELD_SECTION) or ""),
            title=str(entity.get(FIELD_TITLE) or ""),
            mode=str(entity.get(FIELD_MODE) or ""),
            version=str(entity.get(FIELD_VERSION) or ""),
            body=body,
            keywords=str(entity.get(FIELD_KEYWORDS) or ""),
            distance=float(hit.get("distance") or 0.0),
        )


@dataclass
class RetrievalResult:
    """一次检索的账本（``entries`` 与 FTS5 路径同形状）。"""

    entries: list[dict[str, Any]] = field(default_factory=list)
    degraded: str = ""
    retriever: str = ""
    pool: dict[str, int] = field(default_factory=dict)  # 各路 pool 实际条数
    sparse_ms: float = 0.0
    dense_ms: float = 0.0
    embed_ms: float = 0.0
    rerank_ms: float = 0.0
    reranked: int = 0  # 真正被重排模型改过名次的候选数（0 = 没跑或全部失败）
    #: 重排降级标记（与 ``degraded`` 分开：稠密降级后稀疏一路照样可能重排成功，两个降级
    #: 可以同时成立，用单值 ``degraded`` 会互相覆盖）
    rerank_degraded: bool = False
    #: 因**检索器护栏**主动跳过重排的原因（``""`` = 没跳过）。与 ``rerank_degraded`` 区分：
    #: 那是「想排但排失败」，这是「按实测结论压根不排」，两者都不该被当成重排生效。
    rerank_skipped: str = ""
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """只有两路都按预期跑完才算 ok；降级但仍有结果时 ``entries`` 非空。"""
        return not self.degraded

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entries"] = len(self.entries)
        return data


# ---------------------------------------------------------------- 融合


def rank_route(
    hits: Sequence[Hit],
    *,
    version: str = "",
    version_bonus: float = 0.0,
    nav_section_penalty: float = 0.0,
    nav_file_penalty: float = 0.0,
    column_tokens: Sequence[str] = (),
    column_alpha: float = COLUMN_OVERLAP_ALPHA,
) -> dict[int, int]:
    """把一路的命中按「归一化距离 + 后置调整」排序，返回 ``pk -> 1 起的名次``。

    ``column_tokens`` 非空时叠加列权重重叠分（见 ``COLUMN_WEIGHTS``）：Milvus 的 BM25
    丢掉了「命中在哪一列」，这一步把 title/keywords/section/body 的列权重补回来。
    """
    if not hits:
        return {}
    top = max(h.distance for h in hits)
    bottom = min(h.distance for h in hits)
    span = (top - bottom) or 1.0
    scored: list[tuple[float, float, Hit]] = []
    for hit in hits:
        base = (hit.distance - bottom) / span
        if version and hit.version:
            base += version_bonus / SCORE_SPAN_REFERENCE
        if _docs()._is_navigation_section(hit.section):
            base -= nav_section_penalty / SCORE_SPAN_REFERENCE
        if hit.kind == "nav":
            base -= nav_file_penalty / SCORE_SPAN_REFERENCE
        score = base
        if column_tokens:
            overlap = column_overlap_score(hit, column_tokens)
            score = (1.0 - column_alpha) * base + column_alpha * overlap
        scored.append((score, base, hit))
    # pk 兜底保证名次稳定（同分时不让顺序取决于 Milvus 的返回次序）
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].pk))
    return {hit.pk: index + 1 for index, (_score, _base, hit) in enumerate(scored)}


def rrf_fuse(
    routes: Sequence[tuple[str, float, Mapping[int, int]]], *, k: int
) -> dict[int, float]:
    """``score(pk) = Σ weight_route / (k + rank_route(pk))``（只对进了 pool 的块求和）。"""
    fused: dict[int, float] = {}
    for _name, weight, ranks in routes:
        if weight <= 0:
            continue
        for pk, rank in ranks.items():
            fused[pk] = fused.get(pk, 0.0) + float(weight) / (k + rank)
    return fused


# ---------------------------------------------------------------- 检索器


class MilvusRetriever:
    """稀疏/稠密/混合检索。依赖全部可注入，方便单测不碰真库。"""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        settings: Settings | None = None,
        index: MilvusIndex | None = None,
        embedder: Any = None,
        reranker: Any = _RERANKER_UNSET,
        dims: int = 0,
    ) -> None:
        self._settings = settings
        self._config = config or (settings.retrieval if settings is not None else None)
        self._index = index
        self._embedder = embedder
        self._embedder_ready = embedder is not None
        if reranker is _RERANKER_UNSET:
            self._reranker = None
            self._reranker_ready = False
            self._reranker_mode = ""
        else:  # 注入的客户端（含显式 None）直接生效，测试不必碰真实配置
            self._reranker = reranker
            self._reranker_ready = True
            self._reranker_mode = "auto"
        self._dims = int(dims or (settings.embedding.dims if settings is not None else 1024))
        self._lock = threading.Lock()

    # ---- 依赖（懒加载）----

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def config(self) -> RetrievalConfig:
        if self._config is None:
            self._config = self.settings.retrieval
        return self._config

    @property
    def index(self) -> MilvusIndex:
        if self._index is None:
            self._index = get_milvus_index(self.settings)
        return self._index

    @property
    def embedder(self) -> Any:
        """embedding 客户端：构造一次就留着（LRU 查询缓存挂在它身上，每次新建等于没缓存）。"""
        if not self._embedder_ready:
            with self._lock:
                if not self._embedder_ready:
                    try:
                        self._embedder = get_embedding_client(self.settings.embedding)
                    except Exception as exc:  # noqa: BLE001 - 配置坏了也只是稠密降级
                        logger.warning("embedding 客户端构造失败，稠密一路降级：%s", exc)
                        self._embedder = None
                    self._embedder_ready = True
        return self._embedder

    def _reranker_for(self, mode: str) -> Any:
        """按模式取重排客户端（进程内复用；``off`` 不该走到这里）。

        显式 ``mode=api`` 时即使配置写的是 ``off`` 也强制构造一次——评测要能拿
        ``--rerank api`` 覆盖配置；构造不出来（缺 base_url/model）返回 None，由调用方
        按「降级」处理。
        """
        if self._reranker_ready and self._reranker_mode == mode:
            return self._reranker
        with self._lock:
            if self._reranker_ready and self._reranker_mode == mode:
                return self._reranker
            config = self.settings.rerank
            if mode == "api" and config.mode == "off":
                config = replace(config, mode="api")
            try:
                self._reranker = get_reranker(config)
            except Exception as exc:  # noqa: BLE001 - 构造失败也只是重排降级
                logger.warning("rerank 客户端构造失败，保留融合序：%s", exc)
                self._reranker = None
            self._reranker_mode = mode
            self._reranker_ready = True
            return self._reranker

    def close(self) -> None:
        embedder = self._embedder
        self._embedder = None
        self._embedder_ready = False
        if embedder is not None:
            try:
                embedder.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("关闭 embedding 客户端失败：%s", exc)
        reranker = self._reranker
        self._reranker = None
        self._reranker_ready = False
        self._reranker_mode = ""
        if reranker is not None:
            try:
                reranker.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("关闭 rerank 客户端失败：%s", exc)

    # ---- 两路召回 ----

    def _collect(self, result: Any) -> list[Hit]:
        """兼容 Milvus 的 ``SearchResult``（``result[0]`` 是命中表）与测试桩的裸列表。"""
        rows = result
        if isinstance(rows, Sequence) and rows and isinstance(rows[0], Sequence) and not isinstance(rows[0], Mapping):
            rows = rows[0]
        if isinstance(rows, Mapping):  # 桩直接把单条命中给出来
            rows = [rows]
        return [Hit.from_entity(row) for row in (rows or [])]

    def _sparse_search(self, query: str, filter_expr: str, pool_k: int) -> list[Hit]:
        result = self.index.search(
            [expand_query(query)],
            FIELD_SPARSE,
            pool_k,
            filter=filter_expr,
            output_fields=OUTPUT_FIELDS,
        )
        return self._collect(result)

    def _dense_search(self, vector: Sequence[float], filter_expr: str, pool_k: int) -> list[Hit]:
        result = self.index.search(
            [list(vector)],
            FIELD_VECTOR,
            pool_k,
            filter=filter_expr,
            output_fields=OUTPUT_FIELDS,
        )
        return self._collect(result)

    def _embed_query(self, query: str, out: RetrievalResult) -> Sequence[float] | None:
        embedder = self.embedder
        if embedder is None:
            out.degraded = "dense_unavailable"
            return None
        started = time.perf_counter()
        try:
            vector = embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001 - 降级不抛：embedding 挂了还有稀疏一路
            logger.warning("查询向量化失败，降级为稀疏检索：%s", exc)
            out.degraded = "dense_unavailable"
            return None
        finally:
            out.embed_ms = round((time.perf_counter() - started) * 1000, 1)
        if not vector:
            out.degraded = "dense_unavailable"
            return None
        return vector

    # ---- 重排 ----

    def _resolve_rerank(self, rerank: str) -> str:
        if rerank:
            mode = rerank.strip().lower()
        else:
            try:
                mode = str(self.settings.rerank.mode or "auto").strip().lower()
            except Exception:  # noqa: BLE001 - 配置读不到就按未配置（auto 静默）处理
                mode = "auto"
        if mode not in RERANK_MODES:
            raise ValueError(f"未知 rerank 模式：{mode!r}（可选 {'/'.join(RERANK_MODES)}）")
        return mode

    def _apply_rerank(
        self,
        query: str,
        ranked: list[tuple[Hit, int, int | None, float | None]],
        out: RetrievalResult,
        *,
        mode: str,
    ) -> list[tuple[Hit, int, int | None, float | None]]:
        """把融合序的前 ``rerank.top_n`` 条交给重排模型，返回带重排名次的新序。

        返回值每项是 ``(hit, 融合名次, 重排名次或 None, 重排分或 None)``；失败/超时时原样
        返回（只把 ``rerank_degraded`` 置上），**绝不**让检索失败。
        """
        if not ranked:
            return []
        try:
            reranker = self._reranker_for(mode)
            top_n = int(self.settings.rerank.top_n)
            max_chars = int(self.settings.rerank.max_passage_chars)
        except Exception as exc:  # noqa: BLE001 - 配置读不到 = 重排不可用
            logger.warning("rerank 配置读取失败，保留融合序：%s", exc)
            out.rerank_degraded = True
            return list(ranked)
        if reranker is None:
            # auto 未配置是正常状态（静默）；api 未配置说明「以为开了其实没开」，必须留痕
            if mode == "api":
                out.rerank_degraded = True
                logger.warning("rerank.mode=api 但客户端未配置（rerank.base_url/model），保留融合序")
            return list(ranked)

        head = ranked[: max(1, top_n)] if top_n > 0 else ranked
        tail = ranked[len(head) :]
        passages = [_rerank_passage(row[0], max_chars) for row in head]
        tick = time.perf_counter()
        try:
            results = reranker.rerank(query, passages, top_n=len(passages))
        except Exception as exc:  # noqa: BLE001 - 超时/网络/协议异常一律退回融合序
            out.rerank_degraded = True
            logger.warning("rerank 失败，保留融合序：%s", exc)
            return list(ranked)
        finally:
            out.rerank_ms = round((time.perf_counter() - tick) * 1000, 1)
        if not results:
            out.rerank_degraded = True
            return list(ranked)

        reordered: list[tuple[Hit, int, int | None, float | None]] = []
        seen: set[int] = set()
        for index, score in results:
            if not 0 <= int(index) < len(head) or int(index) in seen:
                continue
            seen.add(int(index))
            row = head[int(index)]
            reordered.append((row[0], row[1], len(reordered) + 1, float(score)))
        # 服务端没返回的候选保持融合原序，接在重排结果之后（top_n 之外的也是）
        for index, row in enumerate(head):
            if index not in seen:
                reordered.append((row[0], row[1], None, None))
        out.reranked = len(seen)
        return reordered + list(tail)

    # ---- 主入口 ----

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        mode: str = "",
        version: str = "",
        include_index: bool = False,
        retriever: str = "",
        pool_k: int | None = None,
        rerank: str = "",
    ) -> RetrievalResult:
        config = self.config
        name = (retriever or config.default_retriever or "hybrid").strip().lower()
        if name not in RETRIEVERS:
            raise ValueError(
                f"未知检索器：{name!r}（可选 {'/'.join(RETRIEVERS)}）"
            )
        rerank_mode = self._resolve_rerank(rerank)
        # 护栏：``auto``（也是配置默认）只在稀疏一路上成立。M8 四通道实测：hybrid 上开重排
        # 是净损害——命中率@1 75.98% → 70.39%、MRR 0.819 → 0.784（backend/eval/README.md）。
        # dense 没有实测（M8 没跑这条通道），但它没有可融合的第二路、重排改的就是稠密自己的
        # 序，收益上限低于 sparse，没有理由让它比 sparse 更宽松。
        # 这条结论此前只写在评测报告里靠人自律，现在落到代码：想在这两路上量重排必须显式
        # 传 ``rerank="api"``（夜检通道 C 就是这么跑的），不会因为配置齐全而被静默打开。
        rerank_skipped = ""
        if rerank_mode == "auto" and name != "sparse":
            rerank_skipped = f"retriever_{name}"
            rerank_mode = "off"
        started = time.perf_counter()
        out = RetrievalResult(retriever=name, rerank_skipped=rerank_skipped)
        if rerank_skipped:
            logger.info(
                "重排按检索器护栏跳过：retriever=%s（auto 只在 sparse 上启用，"
                "需要重排请显式传 rerank=api）",
                name,
            )
        # 只为本次会跑的路预置计数键：提前返回时诊断里也能看出「哪一路没跑/跑出 0 条」
        out.pool = {route: 0 for route in ("sparse", "dense") if name in (route, "hybrid")}
        pool_k = int(pool_k or config.pool_k)

        if not query or not query.strip():
            raise ValueError("query 不能为空")

        index = self.index
        if not index.exists():
            out.degraded = "milvus_index_missing"
            out.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            logger.warning("Milvus 索引不存在：%s（先跑 --rebuild）", index.path)
            return out

        doc_filter = _scalar_filter(mode=mode, version=version, include_index=include_index)
        # 约束 1：稠密一路恒 doc-only
        dense_filter = _scalar_filter(mode=mode, version=version, include_index=False)

        sparse_hits: list[Hit] = []
        if name in ("sparse", "hybrid"):
            tick = time.perf_counter()
            try:
                sparse_hits = self._sparse_search(query, doc_filter, pool_k)
            except _MILVUS_ERRORS as exc:
                out.degraded = "milvus_unavailable"
                out.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                logger.warning("Milvus 稀疏检索失败：%s", exc)
                return out
            finally:
                out.sparse_ms = round((time.perf_counter() - tick) * 1000, 1)
            out.pool["sparse"] = len(sparse_hits)

        dense_hits: list[Hit] = []
        if name in ("dense", "hybrid"):
            # ``--no-vectors`` 建的库（CI 稀疏通道、无密钥环境）里每行的 vector 都是零向量，
            # 稠密检索只会按「0 向量的余弦」返回噪声。必须显式降级，不能当真结果给出。
            if index.has_vectors():
                vector = self._embed_query(query, out)
            else:
                vector = None
                out.degraded = "dense_unavailable"
                logger.warning("索引没有稠密向量（--no-vectors 构建）：%s 只走稀疏一路", index.path)
            if vector is None:
                if name == "dense":  # 纯稠密没有后备路径
                    out.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                    return out
            else:
                tick = time.perf_counter()
                try:
                    dense_hits = self._dense_search(vector, dense_filter, pool_k)
                except _MILVUS_ERRORS as exc:
                    out.degraded = "milvus_unavailable"
                    dense_hits = []
                    logger.warning("Milvus 稠密检索失败：%s", exc)
                    if name == "dense":  # 纯稠密没有后备路径
                        out.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                        return out
                finally:
                    out.dense_ms = round((time.perf_counter() - tick) * 1000, 1)
                out.pool["dense"] = len(dense_hits)

        by_pk: dict[int, Hit] = {hit.pk: hit for hit in sparse_hits}
        for hit in dense_hits:
            by_pk.setdefault(hit.pk, hit)

        route_ranks: dict[str, dict[int, int]] = {}
        # 导航文件重降权只在**不显式要清单**时生效（与 FTS5 `_rank` 同口径：include_index=True
        # 时导航页就是答案本身，不能再扣）。dense 一路恒 kind=="doc"，不受影响。
        nav_file_penalty = 0.0 if include_index else float(config.nav_file_penalty)
        # 列权重重排用**原始查询**分词（不是同义词扩展后的串）：扩展只该帮召回，
        # 不该让「靠别名进来」的文档在列覆盖率上得分。稠密一路保持纯语义排序（不打列分）。
        column_tokens = _column_tokens(query)
        if sparse_hits:
            route_ranks["sparse"] = rank_route(
                sparse_hits,
                version=version,
                version_bonus=float(config.version_match_bonus),
                nav_section_penalty=float(config.nav_section_penalty),
                nav_file_penalty=nav_file_penalty,
                column_tokens=column_tokens,
            )
        if dense_hits:
            route_ranks["dense"] = rank_route(
                dense_hits,
                version=version,
                version_bonus=float(config.version_match_bonus),
                nav_section_penalty=float(config.nav_section_penalty),
                nav_file_penalty=nav_file_penalty,
            )
        fused = rrf_fuse(
            [
                ("sparse", float(config.weight_sparse), route_ranks.get("sparse", {})),
                ("dense", float(config.weight_dense), route_ranks.get("dense", {})),
            ],
            k=int(config.rrf_k),
        )
        # 摘要用同一批 jieba 词：它比 FTS5 的单字/双字 token 更贴正文（别再切一次）
        tokens = column_tokens
        # 融合序（名次 1 起）先定下来；重排改的是这个序，改完才做每篇上限与 limit 截断
        # （设计 §8.6：截断在重排之后，否则排得再对也可能被截掉）
        ordered = sorted(fused, key=lambda key: (-fused[key], key))
        ranked: list[tuple[Hit, int, int | None, float | None]] = [
            (by_pk[pk], position, None, None) for position, pk in enumerate(ordered, 1)
        ]
        if rerank_mode != "off":
            ranked = self._apply_rerank(query, ranked, out, mode=rerank_mode)
        entries: list[dict[str, Any]] = []
        for hit, fused_rank, rerank_rank, rerank_score in ranked:
            sources = "+".join(
                sorted(route for route, ranks in route_ranks.items() if hit.pk in ranks)
            )
            entry: dict[str, Any] = {
                "wiki_path": hit.path,
                "kind": hit.kind,
                "section": hit.section,
                "title": hit.title,
                "mode": hit.mode,
                "version": hit.version,
                "score": round(fused[hit.pk], 5),
                "snippet": _docs()._excerpt(hit.body, tokens, SNIPPET_WIDTH),
                "sources": sources,
                "sparse_rank": route_ranks.get("sparse", {}).get(hit.pk),
                "dense_rank": route_ranks.get("dense", {}).get(hit.pk),
                "fused_rank": fused_rank,
            }
            if rerank_rank is not None:
                entry["rerank_rank"] = rerank_rank
                entry["rerank_score"] = rerank_score
            entries.append(entry)
        doc_index = _docs()
        out.entries = doc_index._finalize(entries, max(1, int(limit)), prefix=doc_index.WIKI_DIRNAME)
        out.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return out


# ---------------------------------------------------------------- 进程内单例

_retriever_lock = threading.Lock()
_retriever: MilvusRetriever | None = None


def get_retriever(settings: Settings | None = None) -> MilvusRetriever:
    """进程内单例（embedding 客户端的 LRU 缓存要跨请求复用，不能每次新建）。"""
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = MilvusRetriever(settings=settings)
        return _retriever


def reset_retriever() -> None:
    global _retriever
    with _retriever_lock:
        if _retriever is not None:
            _retriever.close()
        _retriever = None