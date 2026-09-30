"""P4 检索评测：真语料上的检索质量回归测试（CI 门禁）。

与 ``tests/test_doc_index.py`` 的分工：那边用合成语料测「机制」（token 化、降权、小节切片），
跑得快、和语料解耦；这里用 5146 篇真语料测「效果」（命中率 / MRR），慢一些但能发现
「某个常数一调，某类问题就答不上来了」这类退化。

语料是解压产物、不入库，所以跑之前要先 ``python backend/scripts/unpack_doc.py``（CI 里有这一步）；
语料不存在时整个文件跳过，本地不跑也不会红。

M7 起检索只走 Milvus Lite（FTS5 已删），所以还要先有索引库：``cd backend &&
python -m app.agent.milvus_index --rebuild --no-vectors``（CI 里在 pytest 之前建）。
索引库缺失时检索会降级成空结果、命中率恒为 0，此时跳过三条门禁断言而不是把测试判红。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.doc_index import DocIndex
from app.config import load_settings
from eval.run_eval import (
    DEFAULT_DEEP,
    DEFAULT_K,
    DEFAULT_MIN_MRR,
    DEFAULT_MIN_RECALL,
    corpus_paths,
    load_cases,
    run_eval,
    validate_cases,
)

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
_DOC_ROOT = _BACKEND / "doc"
_CORPUS = _DOC_ROOT / "ob_wiki"
_INDEX = load_settings().retrieval.resolve_milvus_path()

pytestmark = pytest.mark.skipif(
    not _CORPUS.is_dir(),
    reason="真语料未解压；先跑 backend/scripts/unpack_doc.py",
)


def _index_ready() -> bool:
    """Milvus Lite 库是否就位（目录存在且有内容；空目录按缺失处理）。"""
    return _INDEX.is_dir() and any(_INDEX.iterdir())


@pytest.fixture(scope="module")
def cases():
    return load_cases()


@pytest.fixture(scope="module")
def report(cases):
    if not _index_ready():
        pytest.skip(
            "Milvus 索引库缺失；先跑 python -m app.agent.milvus_index --rebuild --no-vectors"
        )
    # M7 起 DocIndex 不再就地建库（FTS5 懒建路径已删），索引必须在评测前建好
    index = DocIndex(_DOC_ROOT)
    return run_eval(index, cases, k=DEFAULT_K, deep=DEFAULT_DEEP)


def test_every_expect_pattern_matches_real_docs(cases):
    """用例里的路径子串必须都匹配得到文档：语料升级后别让评测集悄悄过期。"""
    problems = validate_cases(cases, corpus_paths(_DOC_ROOT))
    assert not problems, f"{len(problems)} 个 expect 子串匹配不到语料: {problems[:5]}"


def test_retrieval_recall_meets_gate(report):
    misses = [f"{row['id']}(top1={row['top1']})" for row in report["misses"]]
    assert report["hit_at_k"] >= DEFAULT_MIN_RECALL, (
        f"命中率@{DEFAULT_K} {report['hit_at_k']:.2%} < 门禁 {DEFAULT_MIN_RECALL:.2%}；"
        f"未命中 {len(misses)} 条: {misses}"
    )


def test_retrieval_mrr_meets_gate(report):
    assert report["mrr"] >= DEFAULT_MIN_MRR, (
        f"MRR@{DEFAULT_DEEP} {report['mrr']:.3f} < 门禁 {DEFAULT_MIN_MRR:.3f}；"
        f"排在 {DEFAULT_K} 名之后的有 {[row['id'] for row in report['late']]}"
    )


def test_navigation_pages_never_top_body_answers(report):
    """①②③ 的核心保证：导航页（index.md / 根 README.md）不得顶掉正文答案。

    ``include_index=True`` 的清单类提问是例外——那类问题要的就是目录页。
    """
    offenders = [f"{row['id']}(top1={row['top1']})" for row in report["nav_top1"]]
    assert not offenders, f"导航页抢了正文答案的第一名: {offenders}"