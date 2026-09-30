"""M4 索引构建测试。

全部用注入的**假 embedder**，不打网络、不碰真实 ``backend/doc``；每个用的 data_dir 都在
``tmp_path`` 下（Milvus Lite 是目录级单进程，真实目录会被锁）。
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent import milvus_build as mb
from app.agent.milvus_index import (
    META_SCHEMA_VERSION,
    SCHEMA_VERSION,
    MilvusIndex,
    MilvusUnavailable,
)
from app.config import EmbeddingConfig, RetrievalConfig

DIMS = 8


class FakeEmbedder:
    """确定性假 embedder：同一个文本永远同一向量，不同文本必然不同。"""

    def __init__(self, dims: int = DIMS) -> None:
        self.dims = dims
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([digest[i % len(digest)] / 255.0 for i in range(self.dims)])
        return out

    @property
    def embedded(self) -> int:
        return sum(len(call) for call in self.calls)


@pytest.fixture
def wiki(tmp_path: Path) -> Path:
    root = tmp_path / "ob_wiki"
    root.mkdir()
    (root / "分区表.md").write_text(
        "---\ntitle: 分区表\ndescription: 分区表设计\nkeywords: 分区, 表\n---\n"
        "## 概念\n\n正文 A。\n\n## 用法\n\n正文 B。\n",
        encoding="utf-8",
    )
    (root / "index.md").write_text("# 目录\n\n## 分类列表\n\n清单。\n", encoding="utf-8")
    return root


@pytest.fixture
def config(tmp_path: Path) -> RetrievalConfig:
    return RetrievalConfig(milvus_path=str(tmp_path / "idx.db"))


def make_builder(wiki: Path, config: RetrievalConfig, *, embedder=None, dims: int = DIMS):
    return mb.MilvusBuilder(
        config,
        embedder=embedder if embedder is not None else FakeEmbedder(dims),
        wiki_dir=wiki,
        dims=dims,
        embedding_model="fake-model",
    )


def read_rows(config: RetrievalConfig, data_dir: Path, *, dims: int = DIMS) -> dict[int, dict]:
    index = MilvusIndex(config, dims=dims, path_override=data_dir)
    try:
        rows = index.query(
            filter="pk >= 0",
            output_fields=["pk", "text", "kind", "path", "section", "content_hash", "vector"],
            limit=1000,
        )
    finally:
        index.close()
    return {int(row["pk"]): row for row in rows}


def read_meta(config: RetrievalConfig, data_dir: Path, *, dims: int = DIMS) -> dict[str, str]:
    index = MilvusIndex(config, dims=dims, path_override=data_dir)
    try:
        return index.read_meta()
    finally:
        index.close()


def test_rebuild_after_schema_version_bump_recreates_collections(
    wiki: Path, config: RetrievalConfig
) -> None:
    """SCHEMA_VERSION 升级后 ``--rebuild --no-vectors`` 必须整集重建（旧 schema 不能增量写）。"""
    make_builder(wiki, config).build(mode=mb.MODE_REBUILD, vectors=False)
    data_dir = config.resolve_milvus_path()
    index = MilvusIndex(config, dims=DIMS, path_override=data_dir)
    try:
        meta = index.read_meta()
        assert meta[META_SCHEMA_VERSION] == str(SCHEMA_VERSION)
        meta[META_SCHEMA_VERSION] = "0"  # 假装这是上一版 schema 建出来的库
        index.write_meta(meta)
    finally:
        index.close()

    stats = make_builder(wiki, config).build(mode=mb.MODE_REBUILD, vectors=False)
    assert stats.skipped == 0 and stats.embedded == 0  # 旧行不能复用，整集重写
    assert len(read_rows(config, data_dir)) == 3
    assert read_meta(config, data_dir)[META_SCHEMA_VERSION] == str(SCHEMA_VERSION)


def test_schema_bump_with_vectors_requires_rebuild_vectors(wiki: Path, config: RetrievalConfig) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_REBUILD)
    data_dir = config.resolve_milvus_path()
    index = MilvusIndex(config, dims=DIMS, path_override=data_dir)
    try:
        meta = index.read_meta()
        meta[META_SCHEMA_VERSION] = "0"
        index.write_meta(meta)
    finally:
        index.close()

    with pytest.raises(MilvusUnavailable, match="--rebuild-vectors"):
        make_builder(wiki, config).build(mode=mb.MODE_REBUILD)


# ---------------------------------------------------------------- 扫描


def test_scan_corpus_texts_and_kinds(wiki: Path) -> None:
    payloads = mb.scan_corpus(wiki)
    by_path: dict[str, list[mb.ChunkPayload]] = {}
    for payload in payloads:
        by_path.setdefault(payload.path, []).append(payload)
    assert set(by_path) == {"分区表.md", "index.md"}
    doc = next(p for p in by_path["分区表.md"] if p.section == "概念")
    assert doc.kind == "doc"
    assert doc.title == "分区表 分区表 分区表设计"  # 标题 + 文件名(stem) + description
    assert doc.keywords == "分区 表"
    assert doc.section == "概念"
    # text = 标题×3 + 关键词×2 + 小节 + " | " + 正文
    assert doc.text.startswith("分区表 分区表 分区表设计 分区表 分区表 分区表设计 分区表 分区表 分区表设计")
    assert "分区 表 分区 表 概念 | 正文 A。" in doc.text
    # canonical 每个字段只出现一次，且不含重复前缀
    assert doc.canonical.startswith("分区表 分区表 分区表设计 分区 表 概念 | 正文 A。")
    assert len(doc.content_hash) == 32
    assert doc.content_hash == doc.canonical_hash + doc.text_hash
    assert by_path["index.md"][0].kind == "nav"


def test_scan_corpus_seq_avoids_same_section_collision(tmp_path: Path) -> None:
    root = tmp_path / "ob_wiki"
    root.mkdir()
    # 同一个 H3 路径出现两次会得到相同的小节串，必须靠 seq 区分
    (root / "a.md").write_text(
        "## 甲\n\n### 概览\n\n一。\n\n## 乙\n\n第二节。\n\n## 甲\n\n### 概览\n\n二。\n",
        encoding="utf-8",
    )
    payloads = mb.scan_corpus(root)
    overviews = [p for p in payloads if p.section == "甲 > 概览"]
    assert len(overviews) == 2
    assert len({p.pk for p in overviews}) == 2  # seq 让主键不同
    assert {p.text.split(" | ")[-1] for p in overviews} == {"一。", "二。"}


def test_chunk_pk_is_stable_and_non_negative() -> None:
    pk = mb.chunk_pk("a.md", "概念", 0)
    assert pk == mb.chunk_pk("a.md", "概念", 0)
    assert pk != mb.chunk_pk("a.md", "概念", 1)
    assert 0 <= pk < 2**63


def test_corpus_fingerprint_tracks_body_change(wiki: Path) -> None:
    before = mb.corpus_fingerprint(wiki)
    path = wiki / "分区表.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n补充。\n", encoding="utf-8")
    assert mb.corpus_fingerprint(wiki) != before


def test_missing_wiki_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(MilvusUnavailable, match="文档库不存在"):
        mb.scan_corpus(tmp_path / "nope")


# ---------------------------------------------------------------- 构建


def test_build_fresh_writes_rows_and_meta(wiki: Path, config: RetrievalConfig) -> None:
    builder = make_builder(wiki, config)
    stats = builder.build(mode=mb.MODE_INCREMENTAL)
    assert stats.swapped and stats.verify
    assert stats.chunks == 3 and stats.doc_rows == 2 and stats.nav_rows == 1
    assert stats.embedded == 2  # index.md 是导航页，不算是"打向量"
    assert builder.embedder.embedded == 2

    data_dir = config.resolve_milvus_path()
    rows = read_rows(config, data_dir)
    assert len(rows) == 3
    nav = next(r for r in rows.values() if r["kind"] == "nav")
    doc = next(r for r in rows.values() if r["kind"] == "doc")
    assert all(v == 0.0 for v in nav["vector"])  # 导航页零向量
    assert any(v != 0.0 for v in doc["vector"])
    assert doc["text"].endswith("| 正文 A。") or doc["text"].endswith("| 正文 B。")

    meta = read_meta(config, data_dir)
    assert meta["embedding_model"] == "fake-model"
    assert meta["dims"] == str(DIMS)
    assert meta["analyzer"] == config.analyzer
    assert meta["corpus_fingerprint"] == stats.fingerprint
    assert set(meta) >= {"schema_version", "corpus_fingerprint", "embedding_model", "dims", "analyzer", "built_at"}


def test_incremental_skips_when_fingerprint_unchanged(wiki: Path, config: RetrievalConfig) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    builder = make_builder(wiki, config)
    stats = builder.build(mode=mb.MODE_INCREMENTAL)
    assert stats.skipped_build is True
    assert builder.embedder.embedded == 0


def test_incremental_embeds_only_changed_chunk(wiki: Path, config: RetrievalConfig) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    path = wiki / "分区表.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("正文 A。", "正文 A 改。"), encoding="utf-8"
    )
    builder = make_builder(wiki, config)
    stats = builder.build(mode=mb.MODE_INCREMENTAL)
    assert stats.embedded == 1 and stats.skipped == 2 and stats.pruned == 0
    assert builder.embedder.embedded == 1
    rows = read_rows(config, config.resolve_milvus_path())
    assert len(rows) == 3
    assert any("正文 A 改。" in r["text"] for r in rows.values())


def test_adding_file_embeds_only_new_chunks(wiki: Path, config: RetrievalConfig) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    (wiki / "新页.md").write_text("## 小节\n\n新内容。\n", encoding="utf-8")
    builder = make_builder(wiki, config)
    stats = builder.build(mode=mb.MODE_INCREMENTAL)
    assert stats.embedded == 1 and stats.skipped == 3
    assert len(read_rows(config, config.resolve_milvus_path())) == 4


def test_incremental_prunes_deleted_file(wiki: Path, config: RetrievalConfig) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    (wiki / "index.md").unlink()
    stats = make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    assert stats.pruned == 1
    rows = read_rows(config, config.resolve_milvus_path())
    assert len(rows) == 2


def test_weight_change_rewrites_text_but_reuses_vector(wiki: Path, config: RetrievalConfig) -> None:
    """M6 扫权重的关键性质：改重复次数只重写 BM25 文本，不重打 embedding。"""
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    data_dir = config.resolve_milvus_path()
    before = {pk: row["vector"] for pk, row in read_rows(config, data_dir).items()}

    original = mb.TITLE_REPEAT
    mb.TITLE_REPEAT = original + 2
    try:
        builder = make_builder(wiki, config)
        stats = builder.build(mode=mb.MODE_INCREMENTAL)
    finally:
        mb.TITLE_REPEAT = original

    assert stats.embedded == 0 and builder.embedder.embedded == 0
    assert stats.rewritten == 3 and stats.skipped == 0  # 每块 text 都变了，但都没重打向量
    after = read_rows(config, data_dir)
    for pk, vector in before.items():
        assert after[pk]["vector"] == pytest.approx(vector)  # 向量原样保留
    assert any(row["text"].count("分区表设计") > 3 for row in after.values() if row["kind"] == "doc")


def test_rebuild_mode_ignores_fingerprint_shortcut(wiki: Path, config: RetrievalConfig) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    builder = make_builder(wiki, config)
    stats = builder.build(mode=mb.MODE_REBUILD)
    assert stats.skipped_build is False
    assert stats.skipped == 3 and stats.embedded == 0  # 全部命中旧哈希
    assert len(read_rows(config, config.resolve_milvus_path())) == 3


def test_rebuild_vectors_reembeds_everything_with_new_dims(
    wiki: Path, config: RetrievalConfig
) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    builder = make_builder(wiki, config, dims=12)
    stats = builder.build(mode=mb.MODE_REBUILD_VECTORS, vectors=True)
    assert stats.embedded == 2  # 导航页仍然是零向量
    rows = read_rows(config, config.resolve_milvus_path(), dims=12)
    assert all(len(row["vector"]) == 12 for row in rows.values())
    meta = read_meta(config, config.resolve_milvus_path(), dims=12)
    assert meta["dims"] == "12"


def test_incremental_with_dims_change_tells_user_to_rebuild_vectors(
    wiki: Path, config: RetrievalConfig
) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    builder = make_builder(wiki, config, dims=12)
    with pytest.raises(Exception, match="rebuild-vectors"):
        builder.build(mode=mb.MODE_INCREMENTAL)


def test_no_vectors_build_writes_zero_doc_vectors(wiki: Path, config: RetrievalConfig) -> None:
    builder = mb.MilvusBuilder(config, wiki_dir=wiki, dims=DIMS, embedding_model="")
    stats = builder.build(mode=mb.MODE_INCREMENTAL, vectors=False)
    assert stats.embedded == 0 and stats.doc_rows == 2
    rows = read_rows(config, config.resolve_milvus_path())
    assert all(all(v == 0.0 for v in row["vector"]) for row in rows.values())
    assert read_meta(config, config.resolve_milvus_path())["embedding_model"] == ""
    # 检索层靠这个判据决定「稠密一路要不要降级」（零向量的余弦只是噪声）
    index = MilvusIndex(config, dims=DIMS, path_override=config.resolve_milvus_path())
    try:
        assert index.has_vectors() is False
        assert index.health()["has_vectors"] is False
    finally:
        index.close()


def test_missing_embedder_raises_without_no_vectors(wiki: Path, config: RetrievalConfig) -> None:
    builder = mb.MilvusBuilder(config, wiki_dir=wiki, dims=DIMS)
    with pytest.raises(MilvusUnavailable, match="no-vectors"):
        builder.build(mode=mb.MODE_INCREMENTAL)


def test_locked_data_dir_raises_with_hint(wiki: Path, config: RetrievalConfig, monkeypatch) -> None:
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    (wiki / "另一个.md").write_text("## 小节\n\n内容。\n", encoding="utf-8")

    def boom(path):
        raise MilvusUnavailable("file lock held")

    monkeypatch.setattr(mb, "probe_and_release", boom)
    with pytest.raises(MilvusUnavailable, match="正被其它进程占用"):
        make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)


def test_failed_build_leaves_no_building_dir(wiki: Path, config: RetrievalConfig) -> None:
    builder = make_builder(wiki, config, dims=DIMS)
    # 让 ensure_collections 在 tmp 上直接失败：把 embedding_model 之外的口径弄坏
    make_builder(wiki, config).build(mode=mb.MODE_INCREMENTAL)
    (wiki / "另一个.md").write_text("## 小节\n\n内容。\n", encoding="utf-8")
    bad = make_builder(wiki, config, dims=16)
    with pytest.raises(Exception):
        bad.build(mode=mb.MODE_INCREMENTAL)
    leftovers = list(config.resolve_milvus_path().parent.glob("idx.db.building-*"))
    assert leftovers == []


def test_cli_run_build_no_vectors(wiki: Path, config: RetrievalConfig, tmp_path: Path, capsys) -> None:
    from app.agent.milvus_index import _run_build

    embedding = EmbeddingConfig(base_url="", model="", dims=DIMS)
    settings = SimpleNamespace(retrieval=config, embedding=embedding)
    args = SimpleNamespace(no_vectors=True, limit=0, wiki_dir=str(wiki))
    assert _run_build(settings, mode=mb.MODE_INCREMENTAL, args=args, json_out=True) == 0
    payload = capsys.readouterr().out
    assert '"ok": true' in payload and '"doc_rows": 2' in payload


def test_cli_run_build_without_embedding_reports_error(
    wiki: Path, config: RetrievalConfig, capsys
) -> None:
    from app.agent.milvus_index import _run_build

    embedding = EmbeddingConfig(base_url="", model="", dims=DIMS)
    settings = SimpleNamespace(retrieval=config, embedding=embedding)
    args = SimpleNamespace(no_vectors=False, limit=0, wiki_dir=str(wiki))
    assert _run_build(settings, mode=mb.MODE_INCREMENTAL, args=args, json_out=False) == 1
    assert "no-vectors" in capsys.readouterr().out

def test_h1_only_long_file_is_split_not_truncated(tmp_path: Path) -> None:
    """M7 根治：H1-only 的长文件按 ``MAX_CHUNK_CHARS`` 切块，不再靠 ``max_text_bytes`` 丢后半篇。

    真语料里有 6 篇 8k–40k 字的这种文件（最大 ``obshell/错误码.md`` 40363 字），切块后
    每块都远小于 VARCHAR(max_length=8000)，内容不再被截掉。
    """
    wiki = tmp_path / "ob_wiki"
    wiki.mkdir()
    (wiki / "long.md").write_text("# 巨表\n\n" + "行内容\n" * 4000, encoding="utf-8")
    payloads = mb.scan_corpus(wiki, max_text_bytes=8000)
    assert len(payloads) > 1
    assert not any(p.truncated for p in payloads)
    assert all(len(p.text.encode()) <= 8000 for p in payloads)
    assert {p.section for p in payloads} == {"巨表"}
    # 内容覆盖全文（不再是"只剩前 8000 字节"）
    assert sum(len(p.text) for p in payloads) > len("行内容\n") * 4000 * 0.9
    assert payloads[0].text.startswith(" ".join(["long long"] * mb.TITLE_REPEAT + ["巨表"]) + " | ")


def test_oversized_chunk_is_truncated_to_max_text_bytes(tmp_path: Path) -> None:
    """``max_text_bytes`` 仍是硬守卫：切过块之后单块超过上限，照样截断并记账。"""
    wiki = tmp_path / "ob_wiki"
    wiki.mkdir()
    (wiki / "long.md").write_text("# 巨表\n\n" + "行内容\n" * 50, encoding="utf-8")
    payloads = mb.scan_corpus(wiki, max_text_bytes=64)
    assert payloads and all(p.truncated for p in payloads)
    assert all(len(p.text.encode()) <= 64 for p in payloads)
    assert all(len(p.canonical.encode()) <= 64 for p in payloads)
    assert payloads[0].text.endswith("…") and payloads[0].canonical.endswith("…")


def test_build_reports_truncated_chunks(wiki: Path, config: RetrievalConfig) -> None:
    (wiki / "long.md").write_text("# 巨表\n\n" + "行内容\n" * 4000, encoding="utf-8")
    builder = make_builder(wiki, replace(config, max_text_bytes=64))
    stats = builder.build(mode=mb.MODE_REBUILD, vectors=False)
    assert stats.truncated > 0
    assert f"截断 {stats.truncated}" in stats.summary()
