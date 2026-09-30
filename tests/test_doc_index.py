"""ob_wiki 文档层（app.agent.doc_index）与 search_docs / read_doc 工具的契约测试。

M7 起 doc_index 不再自带索引库：检索交给 ``app.agent.retrieval``（Milvus），本模块只留
「与引擎无关」的部分——路径收敛、frontmatter/keywords 解析、切片 ``_split_chunks``、精读
``read``（直接读文件），以及给工具层的门面。这里用 tmp_path 里的合成小语料固化这些语义；
**检索语义**（命中/降权/模式与版本消歧）在 ``tests/test_retrieval.py`` 里对真 Milvus 固化。
"""
import json
from pathlib import Path

import pytest

from app.agent import doc_index as di
from app.agent.doc_index import (
    DEFAULT_LIMIT,
    DEFAULT_READ_CHARS,
    MAX_CHUNK_CHARS,
    MAX_LIMIT,
    MAX_READ_CHARS,
    DocIndex,
    DocIndexError,
    DocPathError,
    _clean_keywords,
    _clean_title,
    _detect_mode,
    _detect_version,
    _parse_frontmatter,
    _split_chunks,
)

MYSQL_DOC = """\
---
title: MySQL 模式的事务隔离级别
description: OceanBase MySQL 模式支持的事务隔离级别
keywords: 事务,隔离级别,MySQL,OB Cloud 云数据库
---
## 隔离级别设置方法

MySQL 模式下可以通过 SET TRANSACTION ISOLATION LEVEL 设置事务隔离级别，支持读已提交和可重复读。

## 隔离级别行为对比

MySQL 模式的隔离级别行为与原生 MySQL 保持一致。需要注意的是，读已提交（Read Committed）
在 OceanBase MySQL 模式下不会出现幻读，这一点与原生 MySQL 的可重复读（Repeatable Read）等价；
从 V4.2.0 起该行为成为默认值，历史版本需要显式设置。这段较长的说明同时用于验证 read 的截断逻辑。
"""

ORACLE_DOC = """\
---
title: Oracle 模式的事务隔离级别
description: OceanBase Oracle 模式支持的事务隔离级别
keywords: 事务,隔离级别,Oracle
---
## 隔离级别行为对比

Oracle 模式支持读已提交和可串行化，默认读已提交。

## 隔离级别设置方法

Oracle 模式下通过 ALTER SESSION 设置事务隔离级别。
"""

LOCK_DOC = """\
---
title: 锁等待排查
description: 出现锁等待时的排查方法
keywords: 锁等待,行锁,超时
---
## 典型案例

业务报错 lock wait timeout exceeded 时，先查询 GV$OB_LOCKS 找到持有锁的事务，再查 GV$OB_SQL_AUDIT 定位 SQL。

## 相关文档

- 常见等待事件说明
- 分析诊断与决策流程
"""

VERSION_DOC = """\
---
title: OceanBase 数据库企业版
description: 版本发布记录
keywords: 版本,V4.2.5
---
## V4.2.5

V4.2.5 版本新增了租户级资源隔离与更快的合并调度。

## V4.2.4

V4.2.4 版本修复了若干稳定性问题。
"""

INDEX_PAGE = """\
---
title: 事务与隔离级别索引
description: 导航页
keywords: 事务,隔离级别,锁等待,版本
---
# 事务与隔离级别

- [MySQL 模式的事务隔离级别](./mysql.md)
- [Oracle 模式的事务隔离级别](./oracle.md)
- [锁等待排查](./lock.md)
- [版本发布记录](./version.md)
"""


@pytest.fixture
def wiki(tmp_path):
    """合成文档库：导航页 + MySQL/Oracle 同名文档 + 锁等待 + 版本小节。"""
    root = tmp_path / "doc"
    (root / "ob_wiki" / "事务隔离级别").mkdir(parents=True)
    (root / "ob_wiki" / "问题排查").mkdir(parents=True)
    (root / "ob_wiki" / "版本发布记录").mkdir(parents=True)
    (root / "ob_wiki" / "index.md").write_text(INDEX_PAGE, encoding="utf-8")
    (root / "ob_wiki" / "事务隔离级别" / "MySQL 模式的事务隔离级别.md").write_text(MYSQL_DOC, encoding="utf-8")
    (root / "ob_wiki" / "事务隔离级别" / "Oracle 模式的事务隔离级别.md").write_text(ORACLE_DOC, encoding="utf-8")
    (root / "ob_wiki" / "问题排查" / "锁等待排查.md").write_text(LOCK_DOC, encoding="utf-8")
    (root / "ob_wiki" / "版本发布记录" / "OceanBase 数据库企业版.md").write_text(VERSION_DOC, encoding="utf-8")
    return root


@pytest.fixture
def idx(wiki):
    return DocIndex(wiki)


def _long_h1_only_doc(tmp_path: Path, *, paragraphs: int = 200) -> tuple[DocIndex, Path]:
    """只有 H1、没有 H2/H3 的长文档（真语料里有 6 篇，最大 ``obshell/错误码.md`` 40363 字）。"""
    root = tmp_path / "doc" / "ob_wiki"
    root.mkdir(parents=True)
    body = "# 巨表\n\n" + "\n\n".join(f"第 {i} 段：{'长' * 80}" for i in range(paragraphs))
    path = root / "巨表.md"
    path.write_text(body, encoding="utf-8")
    return DocIndex(root.parent), path


# ---- 解析与切片（与建索引共用，不能再依赖索引库）----

def test_parse_frontmatter_splits_meta_and_body():
    meta, body = _parse_frontmatter(MYSQL_DOC)
    assert meta["title"] == "MySQL 模式的事务隔离级别"
    assert meta["keywords"].startswith("事务,隔离级别")
    assert body.lstrip().startswith("## 隔离级别设置方法")


def test_parse_frontmatter_without_block_returns_text():
    meta, body = _parse_frontmatter("## 只有正文\n\n没有 frontmatter。\n")
    assert meta == {}
    assert body.startswith("## 只有正文")


def test_clean_title_and_keywords():
    assert _clean_title("分区表设计-OceanBase") == "分区表设计"
    assert _clean_title("分区表设计") == "分区表设计"
    # 站点标签属于导入残留，不进索引文本
    assert _clean_keywords("事务, OB Cloud 云数据库，隔离级别") == "事务 隔离级别"


def test_detect_mode_and_version():
    assert _detect_mode("ob_wiki/事务隔离级别/（Oracle 模式）事务隔离级别.md") == "Oracle"
    assert _detect_mode("ob_wiki/事务隔离级别/MySQL 租户.md") == "MySQL"
    assert _detect_mode("ob_wiki/普通文档.md") == ""
    assert _detect_version("V4.2.5", "x.md") == "4.2.5"
    assert _detect_version("没有版本", "v3.1.0 更新记录.md") == "3.1.0"


def test_split_chunks_keeps_section_trail():
    chunks = _split_chunks("## 甲\n\n甲正文。\n\n### 乙\n\n乙正文。\n")
    assert [name for name, _ in chunks] == ["甲", "甲 > 乙"]


def test_split_chunks_hard_splits_long_section():
    body = "## 大节\n\n" + "\n\n".join(f"第 {i} 段：{'长' * 80}" for i in range(120))
    chunks = _split_chunks(body)
    assert len(chunks) > 1
    assert all(len(text) <= MAX_CHUNK_CHARS for _, text in chunks)
    assert all(name == "大节" for name, _ in chunks)


def test_split_chunks_splits_h1_only_long_doc(tmp_path):
    """M7 根治：没有 H2/H3 的兜底路径也必须切，否则整篇只能被 max_text_bytes 截掉后半篇。"""
    _, path = _long_h1_only_doc(tmp_path)
    meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    chunks = _split_chunks(body)
    assert len(chunks) > 1
    assert all(name == "巨表" for name, _ in chunks)
    assert all(len(text) <= MAX_CHUNK_CHARS for _, text in chunks)
    # 内容不能再被截断：拼回去要覆盖全文
    assert sum(len(text) for _, text in chunks) > 8000


def test_split_chunks_without_any_heading_uses_body_label():
    chunks = _split_chunks("没有标题的一段正文。\n\n还有第二段。\n")
    assert [name for name, _ in chunks] == ["正文"]


# ---- 精读 read（M7 起直接读文件，不依赖任何索引库）----

def test_read_returns_section_text_and_toc(idx):
    doc = idx.read("事务隔离级别/MySQL 模式的事务隔离级别.md", section="设置方法")
    assert "SET TRANSACTION ISOLATION LEVEL" in doc["text"]
    assert doc["title"] == "MySQL 模式的事务隔离级别"
    assert not doc["truncated"]
    # sections 是整篇的目录，模型据此决定下一节，而不是把整篇拉进上下文
    assert [s["section"] for s in doc["sections"]] == ["隔离级别设置方法", "隔离级别行为对比"]


def test_read_whole_doc_when_no_section(idx):
    """整篇读取返回的是各小节正文的拼接（小节名在 section/sections 字段里，不重复进正文）。"""
    doc = idx.read("事务隔离级别/MySQL 模式的事务隔离级别.md")
    assert "SET TRANSACTION ISOLATION LEVEL" in doc["text"]
    assert "与原生 MySQL 保持一致" in doc["text"]
    assert "隔离级别设置方法" not in doc["text"]
    assert doc["total_chars"] == sum(len(t) for _, t in _split_chunks(_parse_frontmatter(MYSQL_DOC)[1]))


def test_read_title_falls_back_to_filename(tmp_path):
    root = tmp_path / "doc" / "ob_wiki"
    root.mkdir(parents=True)
    (root / "无标题-OceanBase.md").write_text("## 小节\n\n正文。\n", encoding="utf-8")
    assert DocIndex(root.parent).read("无标题-OceanBase.md")["title"] == "无标题"


def test_read_truncates(idx):
    doc = idx.read("事务隔离级别/MySQL 模式的事务隔离级别.md", max_chars=200)
    assert doc["truncated"] is True
    assert len(doc["text"]) <= 200


def test_read_caps_max_chars(idx):
    doc = idx.read("事务隔离级别/MySQL 模式的事务隔离级别.md", max_chars=10**9)
    assert len(doc["text"]) <= MAX_READ_CHARS
    assert DEFAULT_READ_CHARS < MAX_READ_CHARS


def test_read_sees_all_chunks_of_h1_only_doc(tmp_path):
    """兜底切块后 read 也要能读全：整篇 > 8000 字符且不再被当成"截断"。"""
    index, _ = _long_h1_only_doc(tmp_path)
    doc = index.read("巨表.md", max_chars=MAX_READ_CHARS)
    assert doc["total_chars"] > 8000
    assert doc["truncated"] is False
    assert doc["sections"] and all(s["section"] == "巨表" for s in doc["sections"])


def test_read_accepts_ob_wiki_and_doc_prefixes(idx):
    for path in (
        "事务隔离级别/MySQL 模式的事务隔离级别.md",
        "ob_wiki/事务隔离级别/MySQL 模式的事务隔离级别.md",
        "doc/ob_wiki/事务隔离级别/MySQL 模式的事务隔离级别.md",
        "./ob_wiki/事务隔离级别/MySQL 模式的事务隔离级别.md",
    ):
        assert idx.read(path)["path"] == "ob_wiki/事务隔离级别/MySQL 模式的事务隔离级别.md"


def test_read_unknown_section_lists_available(idx):
    with pytest.raises(DocPathError) as err:
        idx.read("问题排查/锁等待排查.md", section="不存在的小节")
    assert "sections" in str(err.value) or "小节" in str(err.value)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.md",
        "/etc/passwd",
        "ob_wiki/../../etc/passwd",
        "C:/windows/system32/config",
        "",
    ],
)
def test_read_rejects_escapes(idx, path):
    with pytest.raises(DocPathError):
        idx.read(path)


def test_read_nonexistent_path(idx):
    with pytest.raises(DocPathError):
        idx.read("问题排查/根本没有这篇.md")


# ---- 检索门面（引擎在 app.agent.retrieval，这里只验证参数转发与降级）----

def _record(monkeypatch, index: DocIndex, *, entries=None) -> dict:
    seen: dict = {}

    def fake(query, engine, **kwargs):
        seen.update(query=query, engine=engine, **kwargs)
        return list(entries or [])

    monkeypatch.setattr(index, "_search_milvus", fake)
    return seen


def test_search_forwards_default_engine(idx, monkeypatch):
    seen = _record(monkeypatch, idx)
    monkeypatch.setattr(di, "_default_retriever", lambda: "hybrid")
    assert idx.search("事务") == []
    assert seen["engine"] == "hybrid"
    assert seen["limit"] == DEFAULT_LIMIT
    assert seen["include_index"] is False


def test_search_forwards_explicit_engine_and_clamps_limit(idx, monkeypatch):
    seen = _record(monkeypatch, idx)
    idx.search("事务", retriever="dense", limit=999)
    assert seen["engine"] == "dense"
    assert seen["limit"] == MAX_LIMIT
    idx.search("事务", limit=0)
    assert seen["limit"] == DEFAULT_LIMIT


def test_search_auto_detects_mode_and_version(idx, monkeypatch):
    seen = _record(monkeypatch, idx)
    idx.search("Oracle 模式的事务隔离级别")
    assert seen["mode"] == "Oracle" and seen["auto_mode"] == "Oracle"
    idx.search("V4.2.5 版本新增了什么")
    assert seen["version"] == "4.2.5" and seen["auto_version"] == "4.2.5"
    idx.search("V4.2.5 版本新增了什么", version="4.2.4")
    assert seen["version"] == "4.2.4" and seen["auto_version"] == ""


def test_search_rejects_empty_query(idx):
    with pytest.raises(DocIndexError):
        idx.search("   ")


def test_search_swallows_retrieval_failure(idx, monkeypatch):
    """检索层任何意外都只记日志、回空结果，不能变成 500。"""
    from app.agent import retrieval as retrieval_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("炸了")

    monkeypatch.setattr(retrieval_module, "get_retriever", boom)
    assert idx.search("事务") == []
    assert idx.last_retrieval is None


# ---- 单例与配置 ----

def test_configure_and_get_index(tmp_path, monkeypatch):
    monkeypatch.setattr(di, "_default", None)
    index = di.configure(tmp_path / "doc")
    assert index.wiki_dir == tmp_path / "doc" / "ob_wiki"
    assert di.get_index() is index, "configure 之后 get_index 必须复用同一实例"
    monkeypatch.setattr(di, "_default", None)
    assert di.get_index().doc_root == di.DEFAULT_DOC_ROOT


def test_default_retriever_falls_back_to_sparse(monkeypatch):
    monkeypatch.setattr(di, "_default_retriever_cache", None)

    def broken():
        raise RuntimeError("没有配置文件")

    monkeypatch.setattr("app.config.load_settings", broken)
    assert di._default_retriever() == "sparse"


# ---- 工具层 ----

@pytest.fixture
def doc_tools(wiki, monkeypatch):
    """把 tools 模块里的 get_index 指到合成语料上，避免碰真文档库与该模块的单例。

    工具层契约与检索引擎无关，只把 Milvus 一路换成固定假命中；read_doc 走的仍是真实现。
    """
    import app.agent.tools as tools_mod
    from app.config import SqlConfig
    from app.tools.ocp.mock import MockOcpClient

    index = DocIndex(wiki)

    def fake_search(query, engine, **kwargs):
        if "隔离级别" not in query:
            return []
        return [{
            "path": "ob_wiki/事务隔离级别/MySQL 模式的事务隔离级别.md",
            "kind": "doc",
            "section": "隔离级别设置方法",
            "title": "MySQL 模式的事务隔离级别",
            "mode": "MySQL",
            "version": "",
            "score": 1.0,
            "snippet": "MySQL 模式下可以通过 SET TRANSACTION ISOLATION LEVEL 设置事务隔离级别。",
        }]

    monkeypatch.setattr(index, "_search_milvus", fake_search)
    monkeypatch.setattr(tools_mod, "get_index", lambda: index)
    built = tools_mod.build_tools(MockOcpClient(), SqlConfig(provider="mock"), send_row_data=True)
    return {t.name: t for t in built}


def test_search_docs_tool_returns_ok_payload(doc_tools):
    out = json.loads(doc_tools["search_docs"].invoke({"query": "MySQL 模式的事务隔离级别"}))
    assert out["ok"] is True
    assert out["hit_count"] == len(out["hits"]) > 0
    assert out["hits"][0]["path"].startswith("ob_wiki/")


def test_search_docs_tool_empty_result_gives_hint(doc_tools):
    out = json.loads(doc_tools["search_docs"].invoke({"query": "zzzqqqxxx"}))
    assert out["ok"] is True
    assert out["hits"] == []
    assert "hint" in out


def test_read_doc_tool_returns_section(doc_tools):
    out = json.loads(doc_tools["read_doc"].invoke(
        {"path": "ob_wiki/问题排查/锁等待排查.md", "section": "典型案例"}
    ))
    assert out["ok"] is True
    assert "GV$OB_LOCKS" in out["text"]


def test_read_doc_tool_rejects_escape_as_not_found(doc_tools):
    out = json.loads(doc_tools["read_doc"].invoke({"path": "../../etc/passwd"}))
    assert out["ok"] is False
    assert out["error_kind"] == "not_found"