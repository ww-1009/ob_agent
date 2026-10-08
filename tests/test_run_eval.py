"""评测器（``backend/eval/run_eval.py``）的机制测试：门禁接线、by_source、重排透传。

与 ``tests/test_retrieval_eval.py`` 的分工：那边用真语料 + 真 Milvus 索引测「效果」，
慢且有环境依赖；这里用桩索引 + 合成语料测「门禁算得对不对」——CI 三通道全靠这些开关，
所以 ``--min-hit1`` / ``--max-p50-ms`` / ``--rerank`` 一旦接错要能立刻发现（不是等 CI 红）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import eval.run_eval as run_eval_mod
from eval.run_eval import _percentile, _source_of, main


# --------------------------------------------------------------------------------------
# 纯函数
# --------------------------------------------------------------------------------------
def test_source_of_passes_through_the_plus_joined_string():
    # 检索层给的 sources 是 "dense+sparse" 串（不是列表）；早期版本把它当列表迭代，
    # 渲染成了 [top1=a+e+p+r+s+s]，这条用例就是那个 bug 的守门人。
    assert _source_of({"sources": "dense+sparse"}) == "dense+sparse"
    assert _source_of({"sources": "sparse"}) == "sparse"


def test_source_of_accepts_list_and_handles_missing():
    assert _source_of({"sources": ["sparse", "dense"]}) == "dense+sparse"
    assert _source_of({"sources": ("sparse",)}) == "sparse"
    assert _source_of({"sources": []}) == "none"
    assert _source_of({}) == "none"
    assert _source_of({"sources": None}) == "none"


def test_percentile_is_nearest_rank_on_sorted_values():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([5.0], 0.5) == 5.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.5) == 3.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.95) == 100.0


# --------------------------------------------------------------------------------------
# main()：桩索引 + 合成语料
# --------------------------------------------------------------------------------------
class StubIndex:
    """按 query 返回固定顺序的路径；``rerank="api"`` 时把顺序倒过来，用来量 rerank_moved。"""

    #: query -> 融合顺序（第一个是融合 top1）
    PLAN = {
        "q1": ["ob_wiki/a.md", "ob_wiki/x.md"],
        "q2": ["ob_wiki/b.md", "ob_wiki/x.md"],
        "q3": ["ob_wiki/x.md", "ob_wiki/c.md"],
        "q4": ["ob_wiki/x.md", "ob_wiki/y.md", "ob_wiki/a.md"],
    }

    def __init__(self, doc_root: Path) -> None:
        self.doc_root = doc_root
        self.calls: list[dict[str, str]] = []
        self.last_retrieval = None

    def search(self, query, *, limit=5, retriever="", rerank="", **kwargs):
        self.calls.append({"query": query, "limit": limit, "retriever": retriever, "rerank": rerank})
        time.sleep(0.005)  # 让延迟门禁有可判的读数（别只靠 0.0 ms 的巧合）
        fused = list(self.PLAN.get(query, []))
        ordered = list(reversed(fused)) if rerank == "api" else fused
        top1_fused = fused[0] if fused else ""
        entries = []
        for fused_rank, path in enumerate(fused, 1):
            entry = {
                "path": path,
                "kind": "doc",
                "title": path,
                "section": "",
                "mode": "",
                "version": "",
                "score": 1.0,
                "snippet": "",
                "sources": "sparse",
                "fused_rank": fused_rank,
            }
            if rerank == "api":
                entry["rerank_rank"] = ordered.index(path) + 1
                entry["rerank_score"] = 1.0 - ordered.index(path) * 0.1
            entries.append(entry)
        entries.sort(key=lambda row: row.get("rerank_rank", row["fused_rank"]))
        self.last_retrieval = type(
            "State", (), {"reranked": len(entries) if rerank == "api" else 0, "rerank_degraded": False}
        )()
        # run_eval 用 fused_rank == 1 取融合 top1；桩里保持原值即可
        _ = top1_fused
        return entries[:limit]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """合成语料 + 用例文件 + 桩索引，返回 (argv 公共部分, 桩索引)。"""
    wiki = tmp_path / "ob_wiki"
    wiki.mkdir()
    for name in ("a.md", "b.md", "c.md", "x.md", "y.md"):
        (wiki / name).write_text(f"# {name}\n", encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    rows = [
        {"id": "q1", "query": "q1", "expect": ["a.md"], "tag": "body"},
        {"id": "q2", "query": "q2", "expect": ["b.md"], "tag": "body"},
        {"id": "q3", "query": "q3", "expect": ["c.md"], "tag": "hard"},
        {"id": "q4", "query": "q4", "expect": ["a.md"], "tag": "hard"},
    ]
    cases.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    index = StubIndex(tmp_path)
    monkeypatch.setattr(run_eval_mod, "DocIndex", lambda doc_root: index)
    common = ["--cases", str(cases), "--doc-root", str(tmp_path)]
    return common, index, tmp_path


def _run(capsys, argv):
    code = main(argv)
    return code, capsys.readouterr().out


def test_baseline_report_has_source_and_tag_breakdowns(sandbox, capsys, tmp_path):
    common, index, _ = sandbox
    json_path = tmp_path / "report.json"
    code, out = _run(
        capsys,
        common
        + ["--retriever", "sparse", "--rerank", "off", "--min-recall", "0.9", "--min-mrr", "0.5",
           "--max-nav-top1", "0", "--json", str(json_path)],
    )
    assert code == 0, out
    assert "PASS" in out and "rerank=off" in out
    assert index.calls and all(call["retriever"] == "sparse" for call in index.calls)
    assert all(call["rerank"] == "off" for call in index.calls)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["hit_at_k"] == 1.0
    assert report["recall_at_1"] == pytest.approx(0.5)
    assert report["by_source"]["sparse"]["count"] == 4
    assert report["by_source"]["sparse"]["hit_at_k"] == 1.0
    assert report["by_tag"]["hard"]["hit_at_1"] == pytest.approx(0.0)  # q3/q4 都不在 @1
    assert report["rerank_moved"] == 0
    assert report["reranked_cases"] == 0
    assert report["rerank_degraded_cases"] == 0


def test_min_hit1_gate_fails_and_passes(sandbox, capsys):
    common, _, _ = sandbox
    # recall@1 = 0.5：0.6 过不去，0.4 过得去
    code, out = _run(capsys, common + ["--min-recall", "0.9", "--min-mrr", "0.5", "--min-hit1", "0.6"])
    assert code == 2
    assert "FAIL" in out and "命中率@1 50.00% < 60.00%" in out
    code, out = _run(capsys, common + ["--min-recall", "0.9", "--min-mrr", "0.5", "--min-hit1", "0.4"])
    assert code == 0, out
    assert "命中率@1 50.00% >= 40.00%" in out


def test_max_p50_gate_fails_and_passes(sandbox, capsys):
    common, _, _ = sandbox
    # 桩每次 search 睡 5ms，所以 P50 ~5ms：1ms 过不去，1000ms 过得去
    code, out = _run(capsys, common + ["--min-recall", "0.9", "--min-mrr", "0.5", "--max-p50-ms", "1"])
    assert code == 2
    assert "P50" in out and "✗" in out
    code, out = _run(capsys, common + ["--min-recall", "0.9", "--min-mrr", "0.5", "--max-p50-ms", "1000"])
    assert code == 0, out
    assert "<= 1000 ms ✓" in out


def test_rerank_api_is_passed_through_and_counted(sandbox, capsys, tmp_path):
    common, index, _ = sandbox
    json_path = tmp_path / "rerank.json"
    code, out = _run(
        capsys,
        common + ["--retriever", "sparse", "--rerank", "api", "--min-recall", "0.4", "--min-mrr", "0.3",
                  "--json", str(json_path)],
    )
    assert code == 0, out
    assert all(call["rerank"] == "api" for call in index.calls)
    assert "重排" in out  # render() 的重排注记
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["rerank"] == "api"
    assert report["reranked_cases"] == 4
    # 桩把顺序倒过来：4 条用例的 top1 都换了人；倒序后 q3/q4 的答案反而升到 @1，
    # 所以 @1 从 0.5 变成 0.5（q1/q2 掉出）——这里只钉住「换人」与重排后的口径。
    assert report["rerank_moved"] == 4
    assert report["recall_at_1"] == pytest.approx(0.5)
    assert all(row["top1_source"] == "sparse" for row in report["cases"] if row["top1"])