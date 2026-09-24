"""ob_wiki 全文检索层（app.agent.doc_index）与 search_docs / read_doc 工具的契约测试。

用 tmp_path 里的**合成小语料**跑，不依赖 backend/doc 下 5000+ 篇真文档（那套只用于实测）：
这里固化的是检索语义（中文双字词命中、模式/版本消歧、导航页默认不参与、路径越界拒绝），
这些点在真语料上实测过的行为见提交说明。
"""
import json

import pytest

from app.agent.doc_index import (
    DocIndex,
    DocIndexMissing,
    DocPathError,
    SCHEMA_VERSION,
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
    """合成文档库：一篇导航页 + MySQL/Oracle 同名文档 + 锁等待 + 带版本小节。"""
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
    # 索引库落在 doc_root 下（默认文件名 ob_wiki.index.db），与真实现一致
    return DocIndex(wiki)


# ---- 建库 ----

def test_ensure_builds_index_and_stats(idx):
    idx.ensure()
    stats = idx.stats()
    assert stats["docs"] == 5
    assert stats["chunks"] >= 6
    assert stats["schema_version"] == SCHEMA_VERSION
    assert idx.db_path.is_file()


def test_ensure_is_idempotent_and_rebuilds_on_corpus_change(idx, wiki):
    idx.ensure()
    before = idx.db_path.stat().st_mtime_ns
    idx.ensure()  # 语料未变 → 不重建
    assert idx.db_path.stat().st_mtime_ns == before
    (wiki / "ob_wiki" / "问题排查" / "合并异常问题排查.md").write_text(
        "---\ntitle: 合并异常问题排查\n---\n## 典型案例\n\n合并卡住时先看 major freeze 进度。\n",
        encoding="utf-8",
    )
    idx.ensure()  # 文件数变化 → 指纹变化 → 重建
    assert idx.stats()["docs"] == 6
    assert idx.search("合并卡住")[0]["path"].endswith("合并异常问题排查.md")


def test_missing_wiki_dir_raises(tmp_path):
    with pytest.raises(DocIndexMissing):
        DocIndex(tmp_path / "nowhere").ensure()


# ---- 检索语义 ----

def test_two_char_chinese_query_hits(idx):
    """「事务」这种双字查询必须命中——trigram 分词器会在这里返回 0 条。"""
    hits = idx.search("事务")
    assert hits
    assert any("事务隔离级别" in h["path"] for h in hits)


def test_mode_in_question_filters_same_named_docs(idx):
    """MySQL/Oracle 有同名文档，提问写了模式就必须只给对应模式那篇。"""
    mysql = idx.search("MySQL 模式的事务隔离级别")
    assert mysql[0]["path"].endswith("MySQL 模式的事务隔离级别.md")
    assert mysql[0]["mode"] == "MySQL"
    assert all("Oracle 模式的" not in h["path"] for h in mysql)

    oracle = idx.search("Oracle 模式的事务隔离级别")
    assert oracle[0]["path"].endswith("Oracle 模式的事务隔离级别.md")
    assert oracle[0]["mode"] == "Oracle"


def test_mode_detection_is_case_insensitive(idx):
    hits = idx.search("oracle 模式的事务隔离级别")
    assert hits[0]["mode"] == "Oracle"


def test_explicit_mode_argument_wins(idx):
    hits = idx.search("事务隔离级别", mode="oracle")
    assert hits
    assert all(h["mode"] in ("Oracle", "") for h in hits)


def test_navigation_pages_excluded_by_default(idx):
    """index.md 是纯链接清单，默认不参与检索；显式打开时才出现。"""
    default = idx.search("事务")
    assert default
    assert all(h["kind"] == "doc" for h in default)
    with_index = idx.search("事务", include_index=True)
    assert any(h["kind"] == "index" for h in with_index)


def test_version_section_gets_bonus_and_filter(idx):
    hits = idx.search("V4.2.5 版本新增了什么")
    assert hits[0]["path"].endswith("OceanBase 数据库企业版.md")
    assert hits[0]["version"] == "4.2.5"
    assert "V4.2.5" in hits[0]["section"]
    assert idx.search("版本新增", version="4.2.5")[0]["version"] == "4.2.5"


def test_result_shape(idx):
    hit = idx.search("锁等待")[0]
    assert set(hit) == {"path", "kind", "section", "title", "mode", "version", "score", "snippet"}
    assert hit["path"].startswith("ob_wiki/")
    assert hit["score"] > 0
    assert len(hit["snippet"]) <= 240


def test_snippet_is_body_text_not_ellipses(idx):
    """命中词落在标题/小节名上时 FTS5 的 snippet() 对正文取不到片段、只剩省略号；
    摘要必须自己在正文里取以命中词为中心的片段，否则模型拿到的是「……」。"""
    hit = idx.search("锁等待")[0]
    assert len(hit["snippet"]) > 40
    assert hit["snippet"].strip("…") != ""
    assert "lock wait" in hit["snippet"].lower() or "GV$OB_LOCKS" in hit["snippet"]


def test_search_rejects_empty_query(idx):
    from app.agent.doc_index import DocIndexError

    with pytest.raises(DocIndexError):
        idx.search("   ")


def test_at_most_two_chunks_per_document(idx):
    hits = idx.search("隔离级别", limit=10)
    paths = [h["path"] for h in hits]
    assert len(paths) == len(set(paths)) or paths.count(paths[0]) <= 2


def test_limit_is_capped(idx):
    assert len(idx.search("事务", limit=999)) <= 20


# ---- 分节读取 ----

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


def test_read_truncates(idx):
    doc = idx.read("事务隔离级别/MySQL 模式的事务隔离级别.md", max_chars=200)
    assert doc["truncated"] is True
    assert len(doc["text"]) <= 200


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


# ---- 工具层 ----

@pytest.fixture
def doc_tools(wiki, monkeypatch):
    """把 tools 模块里的 get_index 指到合成语料上，避免碰真文档库与该模块的单例。"""
    import app.agent.tools as tools_mod
    from app.config import SqlConfig
    from app.tools.ocp.mock import MockOcpClient

    index = DocIndex(wiki)
    monkeypatch.setattr(tools_mod, "get_index", lambda: index)
    built = tools_mod.build_tools(MockOcpClient(), SqlConfig(provider="mock"), send_row_data=True)
    return {t.name: t for t in built}


def test_search_docs_tool_returns_ok_payload(doc_tools):
    out = json.loads(doc_tools["search_docs"].invoke({"query": "MySQL 模式的事务隔离级别"}))
    assert out["ok"] is True
    assert out["hit_count"] == len(out["hits"]) > 0
    assert out["hits"][0]["path"].startswith("ob_wiki/")


def test_search_docs_tool_empty_result_gives_hint(doc_tools):
    # 纯 ASCII 生僻词在中文文档库里必然 0 命中（单个 ASCII 字符会被丢弃，不参与匹配）
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