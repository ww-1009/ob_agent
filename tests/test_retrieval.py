"""M5 Milvus 检索层测试：查询扩展、标量过滤、路内调整、RRF 融合、降级与真库冒烟。

RRF 与降级这些用**注入的桩索引**（可造任意距离、任意失败），形状与真库一致性交给末尾
两条**真 Milvus Lite + 假 embedder** 的集成测试（每个用例的 data_dir 都在 tmp_path 下，
Milvus Lite 是目录级单进程，不能碰真实目录）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent import retrieval as rt
from app.agent.milvus_index import (
    FIELD_KIND,
    FIELD_KEYWORDS,
    FIELD_MODE,
    FIELD_PATH,
    FIELD_PK,
    FIELD_SECTION,
    FIELD_SPARSE,
    FIELD_TEXT,
    FIELD_TITLE,
    FIELD_VECTOR,
    FIELD_VERSION,
    MilvusIndex,
    MilvusUnavailable,
)
from app.config import EmbeddingConfig, RerankConfig, RetrievalConfig, Settings
from test_milvus_build import DIMS, FakeEmbedder, make_builder


class QueryEmbedder(FakeEmbedder):
    """假 embedder + ``embed_query``（检索层按查询走 LRU 那条路径）。"""

    def __init__(self, dims: int = DIMS) -> None:
        super().__init__(dims)
        self.query_calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.embed_documents([text])[0]

    def close(self) -> None:
        pass


def hit(
    pk: int,
    distance: float,
    *,
    path: str | None = None,
    section: str = "小节",
    kind: str = "doc",
    version: str = "",
    mode: str = "",
    body: str = "正文内容",
    keywords: str = "",
) -> dict:
    """造一条 Milvus 形状的命中（``entity`` 里带全部输出字段）。"""
    path = path or f"dir/pk{pk}.md"
    return {
        FIELD_PK: pk,
        "distance": distance,
        "entity": {
            FIELD_PK: pk,
            FIELD_PATH: path,
            FIELD_SECTION: section,
            FIELD_TITLE: f"{path} 标题",
            FIELD_KIND: kind,
            FIELD_MODE: mode,
            FIELD_VERSION: version,
            FIELD_KEYWORDS: keywords,
            FIELD_TEXT: f"标题 标题 标题 | {body}",
        },
    }


def rhit(
    pk: int,
    distance: float,
    *,
    kind: str = "doc",
    section: str = "小节",
    version: str = "",
    mode: str = "",
    title: str = "标题",
    keywords: str = "",
    body: str = "正文内容",
) -> rt.Hit:
    """造一条 ``Hit``（``rank_route``/``rrf_fuse`` 吃对象而不是 Milvus 原始 dict）。"""
    return rt.Hit(
        pk=pk,
        path=f"dir/pk{pk}.md",
        kind=kind,
        section=section,
        title=title,
        mode=mode,
        version=version,
        keywords=keywords,
        body=body,
        distance=distance,
    )


class StubIndex:
    """桩索引：可指定两路命中、可指定失败、可指定索引不存在。"""

    def __init__(
        self,
        *,
        sparse: list[dict] | None = None,
        dense: list[dict] | None = None,
        error: Exception | None = None,
        exists: bool = True,
        vectors: bool = True,
    ) -> None:
        self.sparse = sparse or []
        self.dense = dense or []
        self.error = error
        self._exists = exists
        self._vectors = vectors
        self.calls: list[dict] = []

    @property
    def path(self) -> Path:
        return Path("/tmp/stub.milvus.db")

    def exists(self) -> bool:
        return self._exists

    def has_vectors(self) -> bool:
        return self._vectors

    def search(self, data, anns_field, limit, *, filter="", output_fields=None, **_kw):
        self.calls.append({"data": data, "field": anns_field, "limit": limit, "filter": filter})
        if self.error is not None:
            raise self.error
        rows = self.sparse if anns_field == FIELD_SPARSE else self.dense
        return [rows[:limit]]

    def filters_for(self, field: str) -> list[str]:
        return [c["filter"] for c in self.calls if c["field"] == field]


def make_retriever(
    index,
    *,
    embedder=None,
    retriever_config: RetrievalConfig | None = None,
    reranker=rt._RERANKER_UNSET,
    rerank_config=None,
):
    config = retriever_config or RetrievalConfig()
    # 传一份「未配置 embedding」的 Settings：否则 embedder=None 时产品会读真实 config.yaml
    # 建出真客户端并打网络（测试绝不允许）。rerank 同理：默认未配置 → auto 静默跳过。
    settings = Settings(
        retrieval=config,
        embedding=EmbeddingConfig(),
        rerank=rerank_config or RerankConfig(),
    )
    return rt.MilvusRetriever(
        config, settings=settings, index=index, embedder=embedder, reranker=reranker
    )


# ---------------------------------------------------------------- 纯函数


def test_expand_query_appends_only_matched_synonyms() -> None:
    assert rt.expand_query("OceanBase 是什么") == "OceanBase 是什么"
    expanded = rt.expand_query("怎么排查死锁")
    assert expanded.startswith("怎么排查死锁 ")
    assert "deadlock" in expanded and "锁等待" in expanded
    # 已经在提问里的别名不重复追加
    assert "lock wait" in rt.expand_query("锁等待")
    assert rt.expand_query("锁等待").count("lock wait") == 1


def test_scalar_filter_kind_mode_version() -> None:
    docs_only = rt._scalar_filter(include_index=False)
    assert docs_only == 'kind == "doc"'
    assert rt._scalar_filter(include_index=True) == 'kind in ["doc","nav"]'
    filtered = rt._scalar_filter(mode="oracle", version="V4.2.5", include_index=False)
    assert '(mode == "Oracle" or mode == "")' in filtered
    assert '(version == "" or version like "4.2.5%")' in filtered
    # 引号/反斜杠不能逃出字符串字面量
    escaped = rt._scalar_filter(mode='Or"acle', version='4.2.5"x', include_index=False)
    assert 'mode == "Oracle"' in escaped
    assert 'Or"acle' not in escaped
    assert '4.2.5x%' in escaped


def test_rank_route_prefers_distance_then_adjustments() -> None:
    hits = [
        rhit(1, 0.50, version="4.2.5", section="正文小节"),
        rhit(2, 0.505),
        rhit(3, 0.53),
        rhit(4, 0.55),
    ]
    plain = rt.rank_route(hits)
    assert list(plain) == [4, 3, 2, 1]  # 距离越大越靠前
    assert [plain[pk] for pk in (4, 3, 2, 1)] == [1, 2, 3, 4]
    # 版本命中奖励 = 8/20 的跨度：第 4 名的版本命中行升到第 3 名（但吃不掉整段跨度）
    boosted = rt.rank_route(hits, version="4.2.5", version_bonus=8.0)
    assert boosted[4] == 1 and boosted[1] == 3 and boosted[2] == 4
    # 导航小节 -12/20、导航文件 -40/20：后者一定沉底
    nav = rt.rank_route(
        [rhit(1, 0.50, section="文档明细"), rhit(2, 0.505), rhit(3, 0.55), rhit(9, 0.56, kind="nav")],
        nav_section_penalty=12.0,
        nav_file_penalty=40.0,
    )
    assert nav[3] == 1 and nav[9] == 4 and nav[1] == 3
    assert rt.rank_route([]) == {}


def test_column_tokens_use_jieba_and_drop_noise() -> None:
    """列权重打分用的查询词：jieba 分词，丢掉标点/空白/单个英文字母。"""
    tokens = rt._column_tokens("日志流怎么管理？ 用 4.2.5 / x 版本")
    assert "日志" in tokens and "管理" in tokens
    assert "？" not in tokens and "/" not in tokens and "x" not in tokens
    assert rt._column_tokens("   ") == []
    # 一个词可以同时命中多列（标题 + 正文），keywords 是标题的补充（不叠加）
    both = rhit(1, 0.1, title="日志流", body="日志流管理正文")
    assert rt.column_overlap_score(both, ["日志流"]) == pytest.approx(11.0 / rt.COLUMN_SCORE_SPAN)
    by_keywords = rhit(2, 0.1, title="别的", keywords="日志流", body="")
    assert rt.column_overlap_score(by_keywords, ["日志流"]) == pytest.approx(6.0 / rt.COLUMN_SCORE_SPAN)
    by_section = rhit(3, 0.1, title="别的", section="日志流", body="")
    assert rt.column_overlap_score(by_section, ["日志流"]) == pytest.approx(4.0 / rt.COLUMN_SCORE_SPAN)
    by_body = rhit(4, 0.1, title="别的", section="别的", body="日志流")
    assert rt.column_overlap_score(by_body, ["日志流"]) == pytest.approx(1.0 / rt.COLUMN_SCORE_SPAN)
    assert rt.column_overlap_score(by_body, []) == 0.0


def test_rank_route_column_tokens_promote_title_and_keywords_matches() -> None:
    """列权重重排能翻过距离劣势：标题命中 > 距离压线但不在任何列命中的行（alpha=0.5）。"""
    hits = [
        rhit(1, 0.60, title="别的标题", section="别的", body="完全无关的正文"),
        rhit(2, 0.58, title="日志流管理", section="别的", body="无关正文"),
        rhit(3, 0.50, title="别的", section="别的", body="完全无关的正文"),
    ]
    # 不看列：名次就是距离名次
    assert list(rt.rank_route(hits)) == [1, 2, 3]
    ranks = rt.rank_route(hits, column_tokens=["日志流", "管理"])
    assert ranks[2] == 1 and ranks[1] == 2 and ranks[3] == 3
    # alpha=0 时退回纯距离
    assert list(rt.rank_route(hits, column_tokens=["日志流", "管理"], column_alpha=0.0)) == [1, 2, 3]
    # 没有查询词时不做列打分（稠密一路就是这么调的）
    assert list(rt.rank_route(hits, column_tokens=[])) == [1, 2, 3]


def test_hit_from_entity_reads_keywords_column() -> None:
    parsed = rt.Hit.from_entity(hit(7, 0.9, keywords="分区表,日志流"))
    assert parsed.keywords == "分区表,日志流"
    # 老 schema（没有 keywords 列）取到空串，不报错
    legacy = hit(8, 0.9)
    legacy["entity"].pop(FIELD_KEYWORDS)
    assert rt.Hit.from_entity(legacy).keywords == ""


def test_rrf_fuse_math_and_zero_weight() -> None:
    fused = rt.rrf_fuse([("sparse", 1.0, {1: 1, 2: 2, 3: 3}), ("dense", 1.0, {2: 1, 4: 2})], k=60)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1] == pytest.approx(1 / 61)
    assert fused[4] == pytest.approx(1 / 62)
    assert fused[3] == pytest.approx(1 / 63)
    # 权重 0 的一路不参与
    only_sparse = rt.rrf_fuse([("sparse", 1.0, {1: 1}), ("dense", 0.0, {2: 1})], k=60)
    assert set(only_sparse) == {1}


# ---------------------------------------------------------------- 融合与条目


def test_hybrid_merges_both_routes_by_rrf_score() -> None:
    index = StubIndex(
        sparse=[hit(1, 10.0), hit(2, 9.0), hit(3, 8.0)],
        dense=[hit(2, 0.9), hit(4, 0.8)],
    )
    result = make_retriever(index, embedder=QueryEmbedder()).search(
        "分区表", retriever="hybrid", limit=5
    )
    assert result.degraded == ""
    assert result.retriever == "hybrid"
    assert result.pool == {"sparse": 3, "dense": 2}
    order = [e["path"] for e in result.entries]
    # pk2 两路都在 → 融合分最高；pk1（稀疏第 1）次之；pk4（稠密第 2）再次；pk3 最后
    assert order == [
        "ob_wiki/dir/pk2.md",
        "ob_wiki/dir/pk1.md",
        "ob_wiki/dir/pk4.md",
        "ob_wiki/dir/pk3.md",
    ]
    top = result.entries[0]
    assert top["sources"] == "dense+sparse"
    assert top["sparse_rank"] == 2 and top["dense_rank"] == 1
    assert top["path"] == "ob_wiki/dir/pk2.md"
    assert top["snippet"] == "正文内容"  # 摘要只取 " | " 之后的正文
    assert result.entries[3]["sources"] == "sparse"


def test_dense_route_is_always_doc_only_even_with_include_index() -> None:
    index = StubIndex(sparse=[hit(1, 3.0)], dense=[hit(2, 0.5)])
    retriever = make_retriever(index, embedder=QueryEmbedder())
    retriever.search("分区表", retriever="hybrid", include_index=True)
    assert retriever.index.filters_for(FIELD_SPARSE)[0] == 'kind in ["doc","nav"]'
    assert retriever.index.filters_for(FIELD_VECTOR)[0] == 'kind == "doc"'


def test_mode_and_version_filters_reach_both_routes() -> None:
    index = StubIndex(sparse=[hit(1, 3.0)], dense=[hit(2, 0.5)])
    retriever = make_retriever(index, embedder=QueryEmbedder())
    retriever.search("事务", retriever="hybrid", mode="Oracle", version="4.2.5")
    for expr in retriever.index.filters_for(FIELD_SPARSE) + retriever.index.filters_for(FIELD_VECTOR):
        assert '(mode == "Oracle" or mode == "")' in expr
        assert '(version == "" or version like "4.2.5%")' in expr


def test_same_path_capped_at_two_sections() -> None:
    index = StubIndex(sparse=[hit(pk, 10.0 - pk, path="dir/one.md") for pk in (1, 2, 3, 4)])
    result = make_retriever(index).search("分区表", retriever="sparse", limit=5)
    assert [e["path"] for e in result.entries] == ["ob_wiki/dir/one.md"] * 2
    assert result.entries[0]["score"] > result.entries[1]["score"]


# ---------------------------------------------------------------- 降级


def test_sparse_only_needs_no_embedder() -> None:
    index = StubIndex(sparse=[hit(1, 5.0)])
    result = make_retriever(index, embedder=None).search("分区表", retriever="sparse")
    assert result.degraded == "" and result.ok
    assert [c["field"] for c in index.calls] == [FIELD_SPARSE]


def test_embedding_failure_keeps_sparse_route_in_hybrid() -> None:
    class BrokenEmbedder:
        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("embedding 服务不可用")

        def close(self) -> None:
            pass

    index = StubIndex(sparse=[hit(1, 5.0)], dense=[hit(2, 0.5)])
    result = make_retriever(index, embedder=BrokenEmbedder()).search("分区表", retriever="hybrid")
    assert result.degraded == "dense_unavailable"
    assert not result.ok
    assert [e["path"] for e in result.entries] == ["ob_wiki/dir/pk1.md"]
    assert result.pool["dense"] == 0


def test_pure_dense_without_embedder_returns_empty_degraded() -> None:
    index = StubIndex(dense=[hit(1, 0.5)])
    result = make_retriever(index, embedder=None).search("分区表", retriever="dense")
    assert result.entries == []
    assert result.degraded == "dense_unavailable"


def test_vectorless_index_skips_dense_and_keeps_sparse() -> None:
    """``--no-vectors`` 建出来的库（CI 稀疏通道）不能把零向量当稠密结果返回。"""
    index = StubIndex(sparse=[hit(1, 12.0)], dense=[hit(9, 0.99)], vectors=False)
    embedder = QueryEmbedder()
    retriever = make_retriever(index, embedder=embedder)
    # 纯稠密：直接降级，连 embedding 请求都不发（省配额、也不返回噪声）
    dense = retriever.search("分区表", retriever="dense")
    assert dense.entries == [] and dense.degraded == "dense_unavailable"
    assert embedder.embedded == 0
    # hybrid：保留稀疏一路（RRF 只融合跑起来的那路）
    hybrid = retriever.search("分区表", retriever="hybrid")
    assert hybrid.degraded == "dense_unavailable"
    assert hybrid.pool == {"sparse": 1, "dense": 0}
    assert [e["path"] for e in hybrid.entries] == ["ob_wiki/dir/pk1.md"] and embedder.embedded == 0
    assert [c["field"] for c in index.calls] == [FIELD_SPARSE]  # 稠密那两次都没发


def test_milvus_failure_degrades_without_raising() -> None:
    index = StubIndex(error=MilvusUnavailable("Milvus 没起来"))
    result = make_retriever(index).search("分区表", retriever="sparse")
    assert result.entries == [] and result.degraded == "milvus_unavailable"


def test_missing_index_degrades_without_touching_milvus() -> None:
    index = StubIndex(exists=False)
    result = make_retriever(index).search("分区表", retriever="hybrid")
    assert result.entries == [] and result.degraded == "milvus_index_missing"
    assert index.calls == []  # 连 search 都不该发出去


def test_unknown_retriever_and_empty_query_raise() -> None:
    retriever = make_retriever(StubIndex())
    with pytest.raises(ValueError, match="未知检索器"):
        retriever.search("分区表", retriever="bogus")
    with pytest.raises(ValueError, match="query 不能为空"):
        retriever.search("   ", retriever="sparse")


# ---------------------------------------------------------------- 真库集成


@pytest.fixture
def built(tmp_path: Path):
    wiki = tmp_path / "ob_wiki"
    wiki.mkdir()
    (wiki / "隔离级别.md").write_text(
        "## 事务隔离级别\n\nOceanBase 支持读已提交和可重复读两种隔离级别。\n",
        encoding="utf-8",
    )
    (wiki / "index.md").write_text("# 目录\n\n## 分类列表\n\n隔离级别、事务、锁。\n", encoding="utf-8")
    config = RetrievalConfig(milvus_path=str(tmp_path / "idx.db"))
    builder = make_builder(wiki, config, embedder=QueryEmbedder())
    builder.build(mode="rebuild")
    index = MilvusIndex(config, dims=DIMS, path_override=Path(config.milvus_path))
    retriever = rt.MilvusRetriever(
        config, index=index, embedder=QueryEmbedder(), dims=DIMS, reranker=None
    )
    try:
        yield retriever, config
    finally:
        index.close()


def test_real_index_sparse_search_shape(built) -> None:
    retriever, _config = built
    result = retriever.search("事务隔离级别", retriever="sparse", limit=5)
    assert result.degraded == ""
    assert result.pool["sparse"] >= 1
    top = result.entries[0]
    assert top["path"] == "ob_wiki/隔离级别.md"
    assert top["kind"] == "doc"
    assert "读已提交" in top["snippet"]
    assert top["sources"] == "sparse" and top["sparse_rank"] == 1
    assert top["score"] > 0


def test_real_index_nav_rows_excluded_unless_include_index(built) -> None:
    retriever, _config = built
    only_nav_matches = "分类列表"
    assert retriever.search(only_nav_matches, retriever="sparse").entries == []
    with_nav = retriever.search(only_nav_matches, retriever="sparse", include_index=True)
    assert [e["kind"] for e in with_nav.entries] == ["nav"]
    assert with_nav.entries[0]["path"] == "ob_wiki/index.md"


def test_real_index_dense_and_hybrid_have_no_degradation(built) -> None:
    retriever, _config = built
    dense = retriever.search("隔离级别", retriever="dense", limit=3)
    assert dense.degraded == "" and dense.pool["dense"] >= 1
    assert dense.entries and dense.entries[0]["dense_rank"] == 1
    hybrid = retriever.search("隔离级别", retriever="hybrid", limit=3)
    assert hybrid.degraded == ""
    assert hybrid.pool["sparse"] >= 1 and hybrid.pool["dense"] >= 1
    assert all(e["sources"] for e in hybrid.entries)


# ---------------------------------------------------------------- 后置重排


class StubReranker:
    """桩重排客户端：记录调用、可按脚本给名次、可指定失败。"""

    def __init__(self, order: list[int] | None = None, error: Exception | None = None) -> None:
        self.order = order
        self.error = error
        self.calls: list[dict] = []
        self.closed = False

    def rerank(self, query: str, documents, *, top_n: int = 0):
        self.calls.append({"query": query, "documents": list(documents), "top_n": top_n})
        if self.error is not None:
            raise self.error
        order = self.order if self.order is not None else list(range(len(documents)))
        return [(index, 1.0 - position * 0.01) for position, index in enumerate(order)]

    def close(self) -> None:
        self.closed = True


def _three_hits() -> StubIndex:
    # 归一化距离排序 = pk1, pk2, pk3（融合名次 1/2/3）
    return StubIndex(sparse=[hit(1, 5.0), hit(2, 3.0), hit(3, 1.0)])


def test_rerank_reorders_entries_and_labels_both_ranks() -> None:
    stub = StubReranker(order=[2, 0, 1])
    result = make_retriever(_three_hits(), reranker=stub).search(
        "分区表", retriever="sparse", limit=5
    )
    assert [e["path"] for e in result.entries] == [
        "ob_wiki/dir/pk3.md",
        "ob_wiki/dir/pk1.md",
        "ob_wiki/dir/pk2.md",
    ]
    assert [e["rerank_rank"] for e in result.entries] == [1, 2, 3]
    assert [e["fused_rank"] for e in result.entries] == [3, 1, 2]
    assert result.reranked == 3
    assert result.rerank_degraded is False
    assert result.degraded == "" and result.ok
    assert stub.calls[0]["query"] == "分区表" and stub.calls[0]["top_n"] == 3


def test_rerank_passage_starts_with_title_and_section() -> None:
    stub = StubReranker()
    make_retriever(_three_hits(), reranker=stub).search("分区表", retriever="sparse")
    first = stub.calls[0]["documents"][0]
    assert first.startswith("dir/pk1.md 标题 > 小节")
    assert "正文内容" in first


def test_rerank_happens_before_limit_truncation() -> None:
    """被重排提到第 1 的候选融合名次是 3：先截断的话它会先被丢掉。"""
    stub = StubReranker(order=[2, 0, 1])
    result = make_retriever(_three_hits(), reranker=stub).search(
        "分区表", retriever="sparse", limit=1
    )
    assert [e["path"] for e in result.entries] == ["ob_wiki/dir/pk3.md"]
    assert result.entries[0]["fused_rank"] == 3


def test_rerank_only_touches_top_n_and_keeps_rest_in_fused_order() -> None:
    stub = StubReranker()
    result = make_retriever(
        _three_hits(),
        reranker=stub,
        rerank_config=RerankConfig(base_url="http://x", model="m", top_n=1),
    ).search("分区表", retriever="sparse", limit=5)
    assert len(stub.calls[0]["documents"]) == 1
    assert result.reranked == 1
    assert [e["path"] for e in result.entries] == [
        "ob_wiki/dir/pk1.md",
        "ob_wiki/dir/pk2.md",
        "ob_wiki/dir/pk3.md",
    ]
    assert result.entries[0]["rerank_rank"] == 1
    assert "rerank_rank" not in result.entries[1]


def test_rerank_failure_keeps_fused_order_and_marks_degraded() -> None:
    stub = StubReranker(error=RuntimeError("重排炸了"))
    result = make_retriever(_three_hits(), reranker=stub).search("分区表", retriever="sparse")
    assert [e["path"] for e in result.entries] == [
        "ob_wiki/dir/pk1.md",
        "ob_wiki/dir/pk2.md",
        "ob_wiki/dir/pk3.md",
    ]
    assert result.rerank_degraded is True
    assert result.reranked == 0 and result.rerank_ms >= 0
    assert result.degraded == ""  # 重排降级不该污染检索降级（两路本身是好的）
    assert all("rerank_rank" not in e for e in result.entries)


def test_rerank_off_never_touches_the_client() -> None:
    stub = StubReranker(error=RuntimeError("不该被调用"))
    result = make_retriever(_three_hits(), reranker=stub).search(
        "分区表", retriever="sparse", rerank="off"
    )
    assert stub.calls == []
    assert result.reranked == 0 and result.rerank_degraded is False
    assert all(e["fused_rank"] == i for i, e in enumerate(result.entries, 1))


def test_rerank_api_mode_without_configuration_marks_degraded() -> None:
    result = make_retriever(_three_hits(), reranker=None).search(
        "分区表", retriever="sparse", rerank="api"
    )
    assert result.rerank_degraded is True and result.reranked == 0
    assert len(result.entries) == 3  # 结果照常返回，只是没重排


def test_rerank_auto_mode_without_configuration_is_silent() -> None:
    result = make_retriever(_three_hits(), reranker=None).search(
        "分区表", retriever="sparse", rerank="auto"
    )
    assert result.rerank_degraded is False and result.reranked == 0


def test_unknown_rerank_mode_raises() -> None:
    with pytest.raises(ValueError, match="未知 rerank 模式"):
        make_retriever(_three_hits()).search("分区表", retriever="sparse", rerank="bogus")


def test_close_closes_reranker() -> None:
    stub = StubReranker()
    retriever = make_retriever(_three_hits(), reranker=stub)
    retriever.search("分区表", retriever="sparse")
    retriever.close()
    assert stub.closed is True


# ---------------------------------------------------------------- doc_index 接入


def _doc_index(tmp_path: Path):
    docs = rt._docs()
    return docs.DocIndex(doc_root=tmp_path)


def test_default_retriever_is_read_once_per_process(monkeypatch) -> None:
    """``load_settings()`` 实测 ~5.8 ms/次，引擎选择不能每次检索都重新读配置。"""
    calls: list[int] = []

    def fake_load_settings(*_args, **_kwargs):
        calls.append(1)
        return Settings(retrieval=RetrievalConfig(default_retriever="hybrid"))

    docs = rt._docs()
    monkeypatch.setattr("app.config.load_settings", fake_load_settings)
    monkeypatch.setattr(docs, "_default_retriever_cache", None)
    assert docs._default_retriever() == "hybrid"
    assert docs._default_retriever() == "hybrid"
    assert len(calls) == 1


def test_doc_index_default_engine_is_sparse(built, tmp_path: Path, monkeypatch) -> None:
    """M7 起不传 retriever 就走配置默认 = Milvus 稀疏，FTS5 分支已删除。"""
    retriever, _config = built
    docs = rt._docs()
    monkeypatch.setattr(rt, "get_retriever", lambda settings=None: retriever)
    monkeypatch.setattr(docs, "_default_retriever_cache", None)
    monkeypatch.setattr("app.config.load_settings", lambda *a, **k: Settings(retrieval=RetrievalConfig()))
    index = _doc_index(tmp_path)
    entries = index.search("事务隔离级别", limit=5)
    assert [e["path"] for e in entries] == ["ob_wiki/隔离级别.md"]
    assert index.last_retrieval is not None and index.last_retrieval.retriever == "sparse"


def test_doc_index_routes_to_milvus_when_retriever_given(built, tmp_path: Path, monkeypatch) -> None:
    retriever, _config = built
    monkeypatch.setattr(rt, "get_retriever", lambda settings=None: retriever)
    index = _doc_index(tmp_path)
    entries = index.search("事务隔离级别", limit=5, retriever="sparse")
    assert [e["path"] for e in entries] == ["ob_wiki/隔离级别.md"]
    assert entries[0]["sources"] == "sparse"
    assert index.last_retrieval is not None
    assert index.last_retrieval.retriever == "sparse"
    assert index.last_retrieval.ok


def test_doc_index_survives_retriever_crash(built, tmp_path: Path, monkeypatch) -> None:
    """检索层任何意外都只能变成空结果 + 日志，不能是异常（更不能是 500）。"""

    class Boom:
        def search(self, *args, **kwargs):
            raise RuntimeError("炸了")

    monkeypatch.setattr(rt, "get_retriever", lambda settings=None: Boom())
    assert _doc_index(tmp_path).search("事务隔离级别", retriever="hybrid") == []