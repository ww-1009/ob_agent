"""探针 6：验证 embedding / rerank 端点真实可用（含响应形状与延迟）。

用途：M2（rerank.py）/ M3（embedding.py）落地前，先用真实端点确认
  - OpenAI 兼容 POST {base_url}/embeddings 的请求/响应形状、向量维度是否等于 embedding.dims
  - Jina/Cohere 兼容 POST {base_url}/rerank 的请求/响应形状、结果排序是否合理
  - 端到端延迟量级（决定 rerank.timeout_seconds 与 query timeouts 是否够用）

不写入任何密钥到输出：只打印 base_url / model / 维度 / 状态码 / 延迟。

用法（backend/ 下）：
    ./.venv/bin/python eval/docs/probes/probe6_model_endpoints.py
    ./.venv/bin/python eval/docs/probes/probe6_model_endpoints.py --json   # 另存 .out.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent.parent          # backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import load_settings  # noqa: E402

_TEXT = "OceanBase 集群如何查看租户的资源规格？"
_DOCS = [
    "租户的资源规格通过 DBA_OB_UNITS 视图查看，可以查询 UNIT_ID、CPU 核数与内存大小。",
    "OCP 支持通过浏览器访问，默认端口为 8080，登录后可管理集群。",
    "本文档介绍如何用 obd 命令行部署三副本 OceanBase 集群，包含配置文件示例。",
]


def _embedding(settings) -> dict:
    cfg = settings.embedding
    url = cfg.endpoint_url
    headers = {"Content-Type": "application/json"}
    if cfg.api_key.strip():
        headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"
    out: dict = {"url": url, "model": cfg.model, "expected_dims": cfg.dims}

    # 单条查询：形状 + 维度 + 延迟（这条是 search 热路径）
    t0 = time.perf_counter()
    resp = requests.post(
        url,
        headers=headers,
        json={"model": cfg.model, "input": [_TEXT]},
        timeout=cfg.timeout_seconds,
    )
    out["single_status"] = resp.status_code
    out["single_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if resp.status_code != 200:
        out["error"] = resp.text[:500]
        return out
    body = resp.json()
    out["top_level_keys"] = sorted(body.keys())
    vec = body["data"][0]["embedding"]
    out["single_dims"] = len(vec)
    out["dims_match"] = len(vec) == cfg.dims
    out["vector_head"] = [round(x, 4) for x in vec[:3]]
    out["usage"] = body.get("usage")
    out["item_keys"] = sorted(body["data"][0].keys())

    # 批量：验证顺序、批量上限与建索引吞吐（按配置的 batch_size 实打）
    t0 = time.perf_counter()
    batch = [_TEXT] + [_DOCS[0]] * (max(cfg.batch_size, 1) - 1)
    resp = requests.post(
        url,
        headers=headers,
        json={"model": cfg.model, "input": batch},
        timeout=cfg.timeout_seconds * 4,
    )
    out["batch_status"] = resp.status_code
    out["batch_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["batch_size"] = len(batch)
    if resp.status_code == 200:
        data = resp.json()["data"]
        out["batch_count"] = len(data)
        out["batch_order_ok"] = [d["index"] for d in data] == list(range(len(batch)))
        out["batch_dims"] = sorted({len(d["embedding"]) for d in data})
        out["batch_ms_per_item"] = round(out["batch_ms"] / len(batch), 1)
    else:
        out["batch_error"] = resp.text[:300]
    return out


def _rerank(settings) -> dict:
    cfg = settings.rerank
    url = cfg.endpoint_url
    headers = {"Content-Type": "application/json"}
    if cfg.api_key.strip():
        headers["Authorization"] = f"Bearer {cfg.api_key.strip()}"

    # 用真实候选量级压测（融合后送 top_n 条），而不是 3 条玩具输入
    docs = list(_DOCS)
    while len(docs) < min(cfg.top_n, 15):
        docs.append(f"OceanBase 运维手册第 {len(docs) + 1} 节：OCP 告警配置与巡检项说明。")
    docs = [d[: cfg.max_passage_chars] for d in docs]
    top_n = min(cfg.top_n, len(docs))

    out: dict = {
        "url": url,
        "protocol": cfg.protocol,
        "model": cfg.model,
        "top_n": top_n,
        "passage_chars": cfg.max_passage_chars,
        "timeout": cfg.timeout_seconds,
    }

    if cfg.protocol == "dashscope":
        # 百炼原生形态：input.query / input.documents + parameters
        payload = {
            "model": cfg.model,
            "input": {"query": _TEXT, "documents": docs},
            "parameters": {"top_n": top_n, "return_documents": False},
        }
    else:
        # Jina / Cohere 兼容形态
        payload = {
            "model": cfg.model,
            "query": _TEXT,
            "documents": docs,
            "top_n": top_n,
        }

    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_seconds)
    out["status"] = resp.status_code
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if resp.status_code != 200:
        out["error"] = resp.text[:500]
        return out
    body = resp.json()
    out["top_level_keys"] = sorted(body.keys())
    results = body["output"]["results"] if cfg.protocol == "dashscope" else body.get("results")
    out["result_count"] = len(results) if results is not None else None
    out["top_n_respected"] = out["result_count"] == top_n
    if results:
        out["result_keys"] = sorted(results[0].keys())
        out["ranking"] = [
            {"index": r.get("index"), "score": r.get("relevance_score")} for r in results[:5]
        ]
        # 期望：与查询最相关的第 0 条文档排第一
        out["relevant_first"] = results[0].get("index") == 0
        out["relevance_separated"] = (
            results[0].get("relevance_score", 0) > results[-1].get("relevance_score", 0)
        )
    out["usage"] = body.get("usage")
    return out


def main() -> int:
    settings = load_settings()
    e, r = settings.embedding, settings.rerank
    print(f"embedding.is_configured={e.is_configured} rerank.enabled={r.enabled}")
    if not e.is_configured:
        print("embedding 未配置，跳过端点验证")
    if not r.is_configured:
        print("rerank 未配置（mode=%s），跳过端点验证" % r.mode)

    report: dict = {
        "embedding": _embedding(settings) if e.is_configured else {"skipped": True},
        "rerank": _rerank(settings) if r.is_configured else {"skipped": True},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if "--json" in sys.argv:
        dest = _HERE / "probe6.out.json"
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwritten: {dest}")

    ok = (
        report["embedding"].get("dims_match", report["embedding"].get("skipped"))
        and report["rerank"].get("relevant_first", report["rerank"].get("skipped"))
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())