"""Milvus Lite 索引层（app.agent.milvus_index）契约测试。

用 tmp_path 里的临时 data_dir 跑真实 milvus-lite（每个测试一个目录，互不占锁）。这里固化的是
**集合定义与生命周期契约**：schema 形状（BM25 Function / 不可空向量 / jieba 分析器）、小集合
退回 FLAT、ob_meta 往返与不一致检测、nav 行零向量 + 稠密一路 filter kind=="doc"、
probe_and_release 真的把目录交还、启动校验的硬错/软告警分界。

真语料上的规模与延迟实测见 `backend/eval/docs/probes/probe7.out.json`。
"""
from __future__ import annotations

import json

import pytest
from pymilvus import DataType, FunctionType

from app.agent.milvus_index import (
    FIELD_KIND,
    FIELD_PK,
    ChunkRow,
    MilvusIndex,
    MilvusIndexMissing,
    MilvusSchemaMismatch,
    SCHEMA_VERSION,
    chunks_schema,
    dense_index_type_for,
    meta_entity,
    meta_schema,
    probe_and_release,
    validate_retrieval_config,
)
from app.config import EmbeddingConfig, RerankConfig, RetrievalConfig, Settings

DIM = 8


def make_config(tmp_path, **over) -> RetrievalConfig:
    base = dict(milvus_path=str(tmp_path / "idx.db"))
    base.update(over)
    return RetrievalConfig(**base)


@pytest.fixture
def index(tmp_path):
    idx = MilvusIndex(config=make_config(tmp_path), dims=DIM, embedding_model="test-model")
    yield idx
    idx.close()


def meta_values(*, dims: int = DIM) -> dict[str, str]:
    return {
        "schema_version": str(SCHEMA_VERSION),
        "analyzer": "jieba",
        "dims": str(dims),
        "embedding_model": "test-model",
        "built_at": "2026-09-30T00:00:00",
        "corpus_fingerprint": "100,200,300",
    }


def build(index: MilvusIndex, rows: int = 2) -> MilvusIndex:
    index.ensure_collections(create=True, rows=rows)
    index.write_meta(meta_values())
    return index


def insert_doc_and_nav(index: MilvusIndex) -> None:
    one = [1.0] + [0.0] * (DIM - 1)
    zero = [0.0] * DIM
    rows = [
        ChunkRow(
            pk=1,
            text="事务隔离级别 读已提交 MySQL",
            kind="doc",
            path="事务隔离.md",
            section="隔离级别设置方法",
            title="MySQL 模式的事务隔离级别",
            mode="MySQL",
            version="4.2.5",
            content_hash="h" * 32,
            vector=one,
        ),
        ChunkRow(
            pk=2,
            text="导航页 事务隔离级别 Oracle",
            kind="nav",
            path="导航/事务.md",
            title="事务导航",
            content_hash="n" * 32,
        ),
    ]
    index.client.insert(index.config.collection, [r.to_entity(zero_vector=zero) for r in rows])
    index.client.flush(index.config.collection)


# ---------------------------------------------------------------- schema


def test_chunks_schema_shape():
    schema = chunks_schema(dims=DIM)
    assert [f.name for f in schema.fields] == [
        FIELD_PK,
        "content_hash",
        "text",
        "sparse",
        "vector",
        "kind",
        "path",
        "section",
        "title",
        "mode",
        "version",
    ]
    by_name = {f.name: f for f in schema.fields}
    assert by_name[FIELD_PK].is_primary is True
    assert by_name[FIELD_PK].dtype == DataType.INT64
    # 倒排文本必须开着 jieba 分析器，否则 BM25 对中文退化成整句一个 token
    assert by_name["text"].params["enable_analyzer"] is True
    assert json.loads(by_name["text"].params["analyzer_params"]) == {"type": "jieba"}
    assert by_name["sparse"].dtype == DataType.SPARSE_FLOAT_VECTOR
    assert by_name["vector"].dtype == DataType.FLOAT_VECTOR
    assert by_name["vector"].params["dim"] == DIM
    assert by_name[FIELD_KIND].params["max_length"] == 8
    assert [(f.name, f.input_field_names, f.output_field_names, f.type) for f in schema.functions] == [
        ("bm25", ["text"], ["sparse"], FunctionType.BM25)
    ]


def test_dense_index_type_small_corpus_falls_back_to_flat():
    config = RetrievalConfig(nlist=128)  # IVF 训练点门槛 39×128 = 4992
    assert dense_index_type_for(4, config) == "FLAT"
    assert dense_index_type_for(4991, config) == "FLAT"
    assert dense_index_type_for(4992, config) == "IVF_FLAT"
    # 配置里显式选了 HNSW 时小集合也要尊重它（HNSW 不需要训练点）
    assert dense_index_type_for(312, RetrievalConfig(nlist=8, dense_index_type="HNSW")) == "HNSW"


def test_meta_schema_and_entity_carry_dummy_vector():
    # milvus-lite 不允许没有向量字段的集合，元数据集合必须挂一个占位向量
    schema = meta_schema()
    by_name = {f.name: f for f in schema.fields}
    assert by_name["key"].is_primary is True
    assert by_name["_dummy"].dtype == DataType.FLOAT_VECTOR
    assert by_name["_dummy"].params["dim"] == 2
    assert meta_entity("dims", 8) == {"key": "dims", "value": "8", "_dummy": [0.0, 0.0]}


# ---------------------------------------------------------------- 集合与元数据


def test_create_collections_and_meta_roundtrip(index):
    # 刚建好的集合 ob_meta 必然为空，那是构建中间态，不能当成损坏
    assert index.ensure_collections(create=True, rows=2) == {}
    index.write_meta(meta_values())
    assert index.read_meta() == meta_values()
    assert {index.config.collection, index.config.meta_collection} <= set(index.client.list_collections())
    assert index.has_collections() is True
    # 二次校验通过（schema_version / analyzer / dims 都对得上）
    assert index.ensure_collections(create=False) == meta_values()


def test_missing_collections_require_explicit_create(tmp_path):
    idx = MilvusIndex(config=make_config(tmp_path), dims=DIM)
    try:
        with pytest.raises(MilvusIndexMissing, match="--rebuild"):
            idx.ensure_collections(create=False)
    finally:
        idx.close()


def test_meta_mismatch_is_detected(index):
    build(index)
    index.write_meta(meta_values(dims=DIM + 8))
    with pytest.raises(MilvusSchemaMismatch, match="dims"):
        index.ensure_collections(create=False)

    index.write_meta(meta_values())
    index.write_meta({"schema_version": str(SCHEMA_VERSION + 1)})
    with pytest.raises(MilvusSchemaMismatch, match="schema_version"):
        index.ensure_collections(create=False)

    index.write_meta(meta_values())
    index.write_meta({"analyzer": "chinese"})
    with pytest.raises(MilvusSchemaMismatch, match="analyzer"):
        index.ensure_collections(create=False)


def test_empty_meta_on_existing_collections_is_mismatch(index):
    build(index)
    index.client.delete(index.config.meta_collection, filter='key != ""')
    with pytest.raises(MilvusSchemaMismatch, match="为空"):
        index.ensure_collections(create=False)


# ---------------------------------------------------------------- 行与检索语义


def test_nav_row_zero_vector_and_dense_route_filters_doc(index):
    build(index)
    insert_doc_and_nav(index)

    zero = [0.0] * DIM
    one = [1.0] + [0.0] * (DIM - 1)
    # 检索结果的键就是**主键字段名**（这里是 "pk"；探针脚本里字段叫 "id" 所以写 h["id"]）。
    # 稠密一路恒 filter kind=="doc"：nav 行的零向量（COSINE distance=0）永远不会上浮
    hits = index.search([one], "vector", 5, filter=f'{FIELD_KIND} == "doc"')
    assert [h[FIELD_PK] for h in hits[0]] == [1]
    # 不带过滤时两行都在（这就是 M5 要用 filter 区分两条路径的原因）
    hits = index.search([zero], "vector", 5)
    assert {h[FIELD_PK] for h in hits[0]} == {1, 2}
    # 稀疏一路对 nav 行照常可用：BM25 Function 在 insert 时自动生成了 sparse 向量
    hits = index.search(["事务导航"], "sparse", 5, filter=f'{FIELD_KIND} == "nav"')
    assert [h[FIELD_PK] for h in hits[0]] == [2]


def test_row_requires_vector_or_zero_vector_fallback():
    row = ChunkRow(pk=9, text="t", kind="nav", path="a.md")
    with pytest.raises(Exception, match="缺少向量"):
        row.to_entity()
    entity = row.to_entity(zero_vector=[0.0] * DIM)
    assert entity["vector"] == [0.0] * DIM
    assert entity[FIELD_PK] == 9


def test_stats_and_health(index):
    build(index)
    insert_doc_and_nav(index)
    stats = index.stats()
    assert stats["exists"] is True
    assert stats["rows"] == 2
    assert stats["doc_rows"] == 1
    assert stats["nav_rows"] == 1
    assert stats["dense_index"]["index_type"] == "FLAT"  # 小语料自动退回 FLAT
    assert stats["meta"]["analyzer"] == "jieba"
    assert stats["size_mb"] is not None

    health = index.health()
    assert health["ok"] is True
    assert health["reason"] == "ok"
    assert health["rows"] == 2
    assert health["retrieval_degraded"] is False


def test_stats_and_health_on_missing_dir_do_not_create_it(tmp_path):
    path = tmp_path / "idx.db"
    idx = MilvusIndex(config=make_config(tmp_path), dims=DIM)
    assert idx.stats()["exists"] is False
    health = idx.health()
    assert health["reason"] == "missing"
    assert health["retrieval_degraded"] is True
    # 只读路径探测不该偷偷建出目录（否则「未建索引」会被伪装成「已存在」）
    assert not path.exists()


def test_probe_and_release_hands_the_dir_back(tmp_path):
    path = tmp_path / "idx.db"
    assert probe_and_release(path) == "missing"
    path.mkdir()
    assert probe_and_release(path) == "ok"
    # 第一次探测必须显式 release_server，否则第二次探测会撞上自己留下的 flock
    assert probe_and_release(path) == "ok"


# ---------------------------------------------------------------- 启动校验


def settings_for(tmp_path, *, embedding=None, rerank=None, milvus_path=None) -> Settings:
    return Settings(
        retrieval=RetrievalConfig(milvus_path=milvus_path or str(tmp_path / "idx.db")),
        embedding=embedding or EmbeddingConfig(base_url="https://x/v1", model="m", dims=DIM),
        rerank=rerank or RerankConfig(),
    )


def test_validate_flags_sparse_only_mode(tmp_path):
    warnings = validate_retrieval_config(settings_for(tmp_path, embedding=EmbeddingConfig()))
    assert any("稀疏" in w for w in warnings)


def test_validate_rejects_model_without_base_url(tmp_path):
    with pytest.raises(RuntimeError, match="base_url"):
        validate_retrieval_config(
            settings_for(tmp_path, embedding=EmbeddingConfig(base_url="", model="m", dims=DIM))
        )


def test_validate_rejects_api_rerank_without_endpoint(tmp_path):
    with pytest.raises(RuntimeError, match="rerank"):
        validate_retrieval_config(settings_for(tmp_path, rerank=RerankConfig(mode="api")))


def test_validate_warns_when_dir_missing(tmp_path):
    warnings = validate_retrieval_config(settings_for(tmp_path))
    assert any("不存在" in w for w in warnings)


def test_validate_accepts_built_index(tmp_path):
    idx = MilvusIndex(config=make_config(tmp_path), dims=DIM, embedding_model="m")
    try:
        build(idx)
    finally:
        idx.close()
    assert validate_retrieval_config(settings_for(tmp_path)) == []