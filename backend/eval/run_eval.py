#!/usr/bin/env python
"""P4 检索评测：把「文档检索好不好用」变成可回归的数字。

为什么要有它：``search_docs`` 的排序公式（bm25 + 全词奖励 - 同义词惩罚 + 版本奖励 - 导航惩罚）
是一堆手调常数，改动任何一个都可能悄悄让某类问题变差 —— 只靠几个 e2e 用例看不出。这里用
一批真实提问（``retrieval_cases.jsonl``）跑真语料，产出命中率与 MRR；CI 用它做门禁，
本地用它对比「改前 / 改后」。

指标口径：
- **命中率@k**：前 k 条里出现任一条 ``expect`` 路径子串的用例占比（k 默认 5，与工具默认 limit 一致）。
- **MRR@deep**：第一条命中排名的倒数均值（没命中记 0），衡量「答案排得够不够前」。
- **命中率@1**：答案直接排第一的比例 —— 最接近用户体验的指标。

用例里的 ``expect`` 是**路径子串**而不是精确文件名：同一个问题往往有几篇都算对
（如 MySQL / Oracle 双份文档），判分要认。语料升级后若某个子串一条都匹配不上，
``--strict`` 会直接失败，避免评测集悄悄过期。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:  # 允许直接 `python eval/run_eval.py` 而不用装包
    sys.path.insert(0, str(_BACKEND))

from app.agent.doc_index import DocIndex  # noqa: E402  (需先补 sys.path)

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "retrieval_cases.jsonl"
DEFAULT_DOC_ROOT = _BACKEND / "doc"
DEFAULT_K = 5
DEFAULT_DEEP = 10
# 门禁阈值：基线实测命中率@5 = 82%、MRR@10 = 0.704（50 条用例、5146 篇真语料），
# 阈值取「基线往下留一点」，只在真正退化时失败；改了排序公式就重新量一遍再调这里。
DEFAULT_MIN_RECALL = 0.80
DEFAULT_MIN_MRR = 0.68


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    expect: tuple[str, ...]
    mode: str = ""
    version: str = ""
    include_index: bool = False
    tag: str = ""

    def search_kwargs(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "version": self.version,
            "include_index": self.include_index,
        }


def load_cases(path: Path | str = DEFAULT_CASES) -> list[Case]:
    """读 JSONL 用例；空行与 ``#`` 注释行忽略。行号写进报错，方便改评测集。"""
    text = Path(path).read_text(encoding="utf-8")
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
        missing = [key for key in ("id", "query", "expect") if not item.get(key)]
        if missing:
            raise ValueError(f"{path}:{lineno} 缺少字段 {missing}")
        expect = item["expect"]
        if isinstance(expect, str):
            expect = [expect]
        if not expect:
            raise ValueError(f"{path}:{lineno} expect 不能为空")
        case = Case(
            id=str(item["id"]),
            query=str(item["query"]).strip(),
            expect=tuple(str(p) for p in expect),
            mode=str(item.get("mode") or ""),
            version=str(item.get("version") or ""),
            include_index=bool(item.get("include_index")),
            tag=str(item.get("tag") or ""),
        )
        if case.id in seen:
            raise ValueError(f"{path}:{lineno} 用例 id 重复: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} 里没有用例")
    return cases


def corpus_paths(doc_root: Path | str = DEFAULT_DOC_ROOT, *, wiki_dirname: str = "ob_wiki") -> list[str]:
    """语料里的文档路径，格式与 ``search`` 返回的 ``path`` 一致（``ob_wiki/…``）。"""
    root = Path(doc_root) / wiki_dirname
    return sorted(f"{wiki_dirname}/{p.relative_to(root).as_posix()}" for p in root.rglob("*.md"))


def validate_cases(cases: Sequence[Case], paths: Iterable[str]) -> list[dict[str, str]]:
    """每个 expect 子串至少命中一篇文档，否则判为「过期用例」。"""
    known = list(paths)
    problems: list[dict[str, str]] = []
    for case in cases:
        for pattern in case.expect:
            if not any(pattern in path for path in known):
                problems.append({"id": case.id, "pattern": pattern, "reason": "语料里一条都匹配不上"})
    return problems


def match_rank(hits: Sequence[dict[str, Any]], expect: Sequence[str]) -> tuple[int | None, str]:
    """返回 (首个命中排名 1-based 或 None, 命中的路径)。"""
    for rank, hit in enumerate(hits, 1):
        path = str(hit.get("path") or "")
        if any(pattern in path for pattern in expect):
            return rank, path
    return None, ""


def _percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def run_eval(
    index: DocIndex,
    cases: Sequence[Case],
    *,
    k: int = DEFAULT_K,
    deep: int = DEFAULT_DEEP,
) -> dict[str, Any]:
    """跑一遍评测。``k`` 走生产口径（limit=k），``deep`` 用于 MRR 与「差一点没进前 k」观测。"""
    deep = max(deep, k)
    rows: list[dict[str, Any]] = []
    for case in cases:
        kwargs = case.search_kwargs()
        started = time.perf_counter()
        hits_k = index.search(case.query, limit=k, **kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        hits_deep = hits_k if deep == k else index.search(case.query, limit=deep, **kwargs)
        rank_k, matched_k = match_rank(hits_k, case.expect)
        rank_deep, matched_deep = match_rank(hits_deep, case.expect)
        rows.append(
            {
                "id": case.id,
                "query": case.query,
                "tag": case.tag,
                "include_index": case.include_index,
                "expect": list(case.expect),
                "rank": rank_k,
                "rank_deep": rank_deep,
                "matched": matched_k or matched_deep,
                "top1": str(hits_k[0].get("path") or "") if hits_k else "",
                "top1_score": round(float(hits_k[0].get("score") or 0.0), 2) if hits_k else None,
                "latency_ms": round(latency_ms, 1),
            }
        )

    latencies = [row["latency_ms"] for row in rows]
    ranks = [row["rank_deep"] or 0 for row in rows]
    total = len(rows)
    hits_at_k = sum(1 for row in rows if row["rank"] and row["rank"] <= k)
    hits_at_deep = sum(1 for row in rows if row["rank_deep"])
    # MRR 口径：命中取排名倒数，未命中记 0（顺序敏感，衡量「答案够不够靠前」）
    mrr = sum((1.0 / rank) if rank else 0.0 for rank in ranks) / total if total else 0.0
    report: dict[str, Any] = {
        "k": k,
        "deep": deep,
        "count": total,
        "hit_at_k": round(hits_at_k / total, 4) if total else 0.0,
        "hit_at_deep": round(hits_at_deep / total, 4) if total else 0.0,
        "recall_at_1": round(sum(1 for row in rows if row["rank"] == 1) / total, 4) if total else 0.0,
        "mrr": round(mrr, 4),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.5), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
        },
        "cases": rows,
        "misses": [row for row in rows if not row["rank_deep"]],
        "late": [row for row in rows if row["rank_deep"] and row["rank_deep"] > k],
    }
    tags = sorted({row["tag"] for row in rows if row["tag"]})
    if tags:
        report["by_tag"] = {
            tag: {
                "count": sum(1 for row in rows if row["tag"] == tag),
                "hit_at_k": round(
                    sum(1 for row in rows if row["tag"] == tag and row["rank"] and row["rank"] <= k)
                    / max(1, sum(1 for row in rows if row["tag"] == tag)),
                    4,
                ),
            }
            for tag in tags
        }
    return report


def render(report: dict[str, Any], *, show: int = 8) -> str:
    lines = [
        f"检索评测: {report['count']} 条用例 (k={report['k']}, deep={report['deep']}, 真语料)",
        f"  命中率@1  {report['recall_at_1']:.2%}"
        f"   命中率@{report['k']}  {report['hit_at_k']:.2%}"
        f"   命中率@{report['deep']}  {report['hit_at_deep']:.2%}"
        f"   MRR@{report['deep']}  {report['mrr']:.3f}",
        f"  延迟 ms: 均值 {report['latency_ms']['mean']} / P50 {report['latency_ms']['p50']}"
        f" / P95 {report['latency_ms']['p95']}",
    ]
    for tag, stats in (report.get("by_tag") or {}).items():
        lines.append(f"  [{tag}] {stats['count']} 条, 命中率@{report['k']} {stats['hit_at_k']:.2%}")
    if report["misses"]:
        lines.append(f"  未命中 ({len(report['misses'])}):")
        for row in report["misses"][:show]:
            lines.append(f"    - {row['id']}: {row['query']}")
            lines.append(f"        top1 → {row['top1'] or '(空)'}")
    if report["late"]:
        lines.append(f"  命中但排在 {report['k']} 名之后 ({len(report['late'])}):")
        for row in report["late"][:show]:
            lines.append(f"    - {row['id']}: 第 {row['rank_deep']} 名 → {row['matched']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OceanBase 文档检索评测 / CI 门禁")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="用例 JSONL 路径")
    parser.add_argument("--doc-root", default=str(DEFAULT_DOC_ROOT), help="语料根目录（其下有 ob_wiki/）")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="命中率口径的 top-k")
    parser.add_argument("--deep", type=int, default=DEFAULT_DEEP, help="MRR 口径的深度")
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL, help="命中率@k 门禁")
    parser.add_argument("--min-mrr", type=float, default=DEFAULT_MIN_MRR, help="MRR 门禁")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（排查用）")
    parser.add_argument("--show", type=int, default=8, help="明细里最多列几条")
    parser.add_argument("--json", dest="json_path", default="", help="把完整报告写到该文件")
    parser.add_argument("--strict", action="store_true", help="expect 子串匹配不到任何文档时直接失败")
    parser.add_argument("--rebuild", action="store_true", help="先重建索引（语料指纹变了会自动重建）")
    args = parser.parse_args(argv)

    doc_root = Path(args.doc_root)
    if not (doc_root / "ob_wiki").is_dir():
        print(f"找不到语料目录 {doc_root / 'ob_wiki'}；先跑 scripts/unpack_doc.py", file=sys.stderr)
        return 3

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    problems = validate_cases(cases, corpus_paths(doc_root))
    if problems:
        head = "; ".join(f"{p['id']}→{p['pattern']}" for p in problems[:5])
        print(f"评测集里有 {len(problems)} 个子串匹配不到语料（用例过期）: {head}", file=sys.stderr)
        if args.strict:
            return 3

    index = DocIndex(doc_root)
    if args.rebuild:
        index.ensure(force=True)

    started = time.perf_counter()
    report = run_eval(index, cases, k=args.k, deep=args.deep)
    total_s = time.perf_counter() - started
    report["problems"] = problems
    report["elapsed_s"] = round(total_s, 1)

    print(render(report, show=args.show))
    recall_ok = report["hit_at_k"] >= args.min_recall
    mrr_ok = report["mrr"] >= args.min_mrr
    print(
        f"  门禁: 命中率@{args.k} {report['hit_at_k']:.2%} "
        f"{'>=' if recall_ok else '<'} {args.min_recall:.2%} {'✓' if recall_ok else '✗'}"
        f"   MRR {report['mrr']:.3f} "
        f"{'>=' if mrr_ok else '<'} {args.min_mrr:.3f} {'✓' if mrr_ok else '✗'}"
    )
    print(f"  {'PASS' if recall_ok and mrr_ok else 'FAIL'}（{total_s:.1f}s）")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0 if recall_ok and mrr_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())