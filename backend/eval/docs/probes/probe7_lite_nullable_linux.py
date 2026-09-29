"""probe7：milvus-lite 3.2.1 / Linux x86_64 上的实施前未知项。

probe1–5 在 macOS arm64 上验证了 BM25(jieba) 可用、capacity 与延迟；probe6 验了模型端点。
本探针只问「落地到 backend/app/agent/milvus_index.py 前必须知道答案」的问题：

1. FLOAT_VECTOR 能不能留空（nav 行不写稠密向量）？不能的话零向量代价多大？
2. VARCHAR 的 max_length 单位是字节还是字符（决定正文 8000 守卫怎么算）？
3. jieba 分析器对字面量查询（ob_query_timeout / GV$OB_UNITS / ORA-00942 / V4.2.5）好不好使？
4. 过滤表达式（kind/path/mode/version）哪些写法在 Lite 上能过？
5. 同一 data_dir 多进程打开的真实报错文本（run.sh/测试要据此诊断）。
6. 内建 hybrid_search 与自研 RRF 的排序是否一致、延迟差多少。
7. ob_meta 这类小集合能否与主集合同库共存、query/filter 是否可用。
8. Linux 上 3k 块的稀疏/稠密/hybrid 延迟量级（M6 延迟门禁的目标平台）。

用法：backend/.venv/bin/python backend/eval/docs/probes/probe7_lite_nullable_linux.py
输出：同目录 probe7.out.json（stdout 也打印一份）

环境变量：
- `PROBE7_DIR`：f1–f8 的 data_dir（默认 /tmp/milvus_spike/data7.db）
- `PROBE7_BIG_DIR`：f9 的 data_dir（默认 /tmp/milvus_spike/data9.db，每次重建）
- `PROBE7_ONLY=9`：只跑第 9 段（干净进程测 25k 延迟）并**合并**进已有 probe7.out.json
- `PROBE7_BIG=1`：完整运行时追加第 9 段（否则只跑 1–8）

注意：量延迟时机器上不要有别的重活（25k 构建/编译/另一个探针）。2 vCPU 上并发跑两个
Milvus 进程会把 P50 从 ~80ms 推到 ~800ms。
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

R: dict = {}
DIR = os.environ.get("PROBE7_DIR", "/tmp/milvus_spike/data7.db")
# f9（25k 延迟）单独一个 data_dir：进程内已有 f1–f8 的集合时 RSS 会到 1.4GB，
# 在 2 vCPU/3.9GB 的机器上会把延迟推高一个数量级（实测 sparse 80ms → 800ms）。
DIR_BIG = os.environ.get("PROBE7_BIG_DIR", "/tmp/milvus_spike/data9.db")
DIM = 1024
JIEBA = {"type": "jieba"}


def rec(name, fn):
    t0 = time.time()
    try:
        R[name] = {"ok": True, "result": fn(), "seconds": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001 - 探针要记录而不是抛出
        R[name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:400],
            "seconds": round(time.time() - t0, 2),
        }


def dirsize(p: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / 1e6, 1)


def schema(text_max=8000, dense=True, extra=("kind", "path", "mode", "version")):
    s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field("id", DataType.INT64, is_primary=True)
    s.add_field("text", DataType.VARCHAR, max_length=text_max, enable_analyzer=True, analyzer_params=JIEBA)
    s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    for f in extra:
        s.add_field(f, DataType.VARCHAR, max_length=512 if f == "path" else 16)
    if dense:
        s.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    s.add_function(
        Function(name="bm25", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"])
    )
    return s


def index_params(dense=True, dense_type="IVF_FLAT", extra=("kind", "path", "mode", "version")):
    ip = MilvusClient.prepare_index_params() if hasattr(MilvusClient, "prepare_index_params") else None
    if ip is None:  # pymilvus 3.x：索引参数用同一个 client 实例准备
        raise RuntimeError("MilvusClient.prepare_index_params 不存在")
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for f in extra:
        ip.add_index(f, index_type="INVERTED")
    if dense:
        ip.add_index("vector", index_type=dense_type, metric_type="COSINE", params={"nlist": 128})
    return ip


def ip_for(client, dense=True, dense_type="IVF_FLAT", extra=("kind", "path", "mode", "version")):
    ip = client.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for f in extra:
        ip.add_index(f, index_type="INVERTED")
    if dense:
        ip.add_index("vector", index_type=dense_type, metric_type="COSINE", params={"nlist": 128})
    return ip


def settle(client, name, seconds=6.0, fields=("sparse",), warm_data=None):
    """等索引真正建完 + 预热。

    踩过的坑：`describe_index(...).pending_index_rows == 0` 并不代表后台建索引已经结束，
    紧接着测出来的延迟会虚高一个数量级（3k 集合稠密 276ms vs 预热后 10ms）。
    另一个坑：稀疏一路的 query 必须是**字符串或 dict**，传 float 向量会报
    `Sparse search query must be a dict or string, got list`——所以预热数据要按字段选。
    """
    for f in fields:
        ready(client, name, 0, f)
    time.sleep(seconds)
    for f in fields:
        if f == "sparse":
            data = warm_data if isinstance(warm_data, str) else "预热查询"
        else:
            data = warm_data if warm_data is not None else "预热查询"
        try:
            client.search(name, data=[data], anns_field=f, limit=1)
        except Exception:  # noqa: BLE001
            pass


def ready(client, name, expect, field="sparse", tries=90):
    for _ in range(tries):
        try:
            pending = client.describe_index(name, field).get("pending_index_rows", 0)
        except Exception:  # noqa: BLE001
            pending = 0
        rows = client.get_collection_stats(name).get("row_count", 0)
        if pending == 0 and rows >= expect:
            return True
        time.sleep(1)
    return False


def zero_vec():
    return [0.0] * DIM


def one_hot(i):
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def rng_vec(seed: int):
    import numpy as np

    return np.random.default_rng(seed).random(DIM, dtype=np.float32).tolist()


def main() -> None:
    shutil.rmtree(DIR, ignore_errors=True)
    Path(DIR).parent.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(DIR)

    # ---- 1. FLOAT_VECTOR 可否留空 ----
    def f1():
        name = "nullable"
        client.create_collection(name, schema=schema(), index_params=ip_for(client))
        out = {}
        rows = [
            {"id": 1, "text": "锁等待 SQL 怎么看", "kind": "doc", "path": "a.md", "mode": "", "version": "", "vector": one_hot(0)},
            {"id": 2, "text": "备份恢复总览", "kind": "doc", "path": "b.md", "mode": "", "version": "", "vector": one_hot(1)},
        ]
        client.insert(name, rows)
        # (a) 整行省略 vector 字段
        try:
            client.insert(name, [{"id": 3, "text": "术语表 导航页", "kind": "nav", "path": "index.md", "mode": "", "version": ""}])
            out["omit_field"] = "accepted"
        except Exception as exc:  # noqa: BLE001
            out["omit_field"] = f"{type(exc).__name__}: {exc}"[:200]
        # (b) 显式 None
        try:
            client.insert(name, [{"id": 4, "text": "FAQ 导航页", "kind": "nav", "path": "faq.md", "mode": "", "version": "", "vector": None}])
            out["explicit_none"] = "accepted"
        except Exception as exc:  # noqa: BLE001
            out["explicit_none"] = f"{type(exc).__name__}: {exc}"[:200]
        # (c) 零向量兜底
        try:
            client.insert(name, [{"id": 5, "text": "简介 导航页", "kind": "nav", "path": "intro.md", "mode": "", "version": "", "vector": zero_vec()}])
            out["zero_vector"] = "accepted"
        except Exception as exc:  # noqa: BLE001
            out["zero_vector"] = f"{type(exc).__name__}: {exc}"[:200]
        time.sleep(2)
        out["row_count"] = client.get_collection_stats(name).get("row_count")
        # 零向量行会不会污染稠密一路？COSINE 对零向量怎么算？
        try:
            hits = client.search(name, data=[one_hot(0)], anns_field="vector", limit=5, output_fields=["kind", "path"])
            out["dense_top5"] = [(h["id"], h["entity"]["kind"], round(float(h["distance"]), 4)) for h in hits[0]]
        except Exception as exc:  # noqa: BLE001
            out["dense_top5"] = f"{type(exc).__name__}: {exc}"[:200]
        # 稠密一路带 kind 过滤：nav 行被排除后还剩几条
        try:
            hits = client.search(
                name, data=[one_hot(0)], anns_field="vector", filter='kind == "doc"', limit=5, output_fields=["path"]
            )
            out["dense_filtered_doc_only"] = [(h["id"], h["entity"]["path"]) for h in hits[0]]
        except Exception as exc:  # noqa: BLE001
            out["dense_filtered_doc_only"] = f"{type(exc).__name__}: {exc}"[:200]
        out["nullable_dense"] = "omit_field" in out and out["omit_field"] == "accepted"
        return out

    # ---- 2. VARCHAR max_length 单位 ----
    def f2():
        name = "maxlen"
        client.create_collection(name, schema=schema(text_max=100, dense=False, extra=()), index_params=ip_for(client, dense=False, extra=()))
        out = {}
        cases = {
            "ascii_90_bytes": "a" * 90,
            "cjk_40_chars_120_bytes": "中" * 40,
            "cjk_60_chars_180_bytes": "中" * 60,
            "cjk_100_chars_300_bytes": "中" * 100,
        }
        for i, (label, text) in enumerate(cases.items(), start=1):
            try:
                client.insert(name, [{"id": i, "text": text}])
                out[label] = "accepted"
            except Exception as exc:  # noqa: BLE001
                out[label] = f"{type(exc).__name__}: {exc}"[:160]
        out["max_length_unit"] = "chars" if out.get("cjk_100_chars_300_bytes") == "accepted" else "bytes"
        out["note"] = "100 = max_length；CJK 40 字=120 字节，若按字节算应当直接失败"
        return out

    # ---- 3. jieba 分析器对字面量查询 ----
    def f3():
        name = "jieba"
        client.create_collection(name, schema=schema(dense=False, extra=()), index_params=ip_for(client, dense=False, extra=()))
        docs = [
            {"id": 1, "text": "ob_query_timeout 设置查询超时时间 系统变量"},
            {"id": 2, "text": "GV$OB_UNITS 视图展示资源单元分布"},
            {"id": 3, "text": "ORA-00942 表或视图不存在"},
            {"id": 4, "text": "OceanBase 4.2.5 版本新增特性说明"},
            {"id": 5, "text": "如何查看锁等待的 SQL 语句"},
            {"id": 6, "text": "V$SYSSTAT 统计项含义与常用事件"},
            {"id": 7, "text": "GV$SYSSTAT 统计项含义"},
            {"id": 8, "text": "备份恢复总览与物理备份"},
        ]
        client.insert(name, docs)
        ready(client, name, len(docs))
        out = {}
        for q in ["ob_query_timeout", "GV$OB_UNITS", "ORA-00942", "00942", "4.2.5", "V$SYSSTAT", "锁等待的 SQL", "备份恢复"]:
            hits = client.search(name, data=[q], anns_field="sparse", limit=3, output_fields=["text"])
            out[q] = [(h["id"], round(float(h["distance"]), 3)) for h in hits[0]]
        return out

    # ---- 4. 过滤表达式 ----
    def f4():
        name = "filters"
        client.create_collection(name, schema=schema(dense=False), index_params=ip_for(client, dense=False))
        client.insert(
            name,
            [
                {"id": 1, "text": "锁等待 排查", "kind": "doc", "path": "a/分区表设计.md", "mode": "MySQL", "version": "4.2.5"},
                {"id": 2, "text": "锁等待 排查 Oracle", "kind": "doc", "path": "a/b.md", "mode": "Oracle", "version": "3.2.4"},
                {"id": 3, "text": "锁等待 导航", "kind": "nav", "path": "a/index.md", "mode": "", "version": ""},
            ],
        )
        ready(client, name, 3)
        exprs = [
            'kind == "doc"',
            'kind in ["doc", "nav"]',
            'kind != "nav"',
            'path like "%分区表设计.md"',
            'path like "a/%"',
            '(mode == "Oracle" or mode == "")',
            '(version == "" or version like "4.2.5%")',
            'kind == "doc" and version == "4.2.5"',
            'id in [1, 3]',
        ]
        out = {}
        for e in exprs:
            try:
                hits = client.search(name, data=["锁等待"], anns_field="sparse", filter=e, limit=5)
                out[e] = [h["id"] for h in hits[0]]
            except Exception as exc:  # noqa: BLE001
                out[e] = f"{type(exc).__name__}: {exc}"[:160]
        return out

    # ---- 5. 多进程文件锁 ----
    def f5():
        code = (
            "import sys;from pymilvus import MilvusClient\n"
            f"c=MilvusClient({DIR!r})\nprint('child_opened', sorted(c.list_collections()))"
        )

        def child():
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip()[:200],
                "stderr": proc.stderr.strip()[-200:],
            }

        while_open = child()
        client.close()
        after_close = child()
        # 显式释放：pymilvus 里没有 release_server 调用，close() 不会放开 LOCK；
        # 手动调 milvus_lite 的 ServerManager 才是同一进程内交接目录的唯一办法。
        from milvus_lite.server_manager import server_manager_instance  # noqa: PLC0415

        server_manager_instance.release_server(DIR)
        after_release = child()
        try:
            reopened = sorted(MilvusClient(DIR).list_collections())
        except Exception as exc:  # noqa: BLE001
            reopened = f"{type(exc).__name__}: {exc}"[:120]
        return {
            "while_open": while_open,
            "after_close": after_close,
            "after_release_server": after_release,
            "in_process_reopen": reopened,
        }

    def f6():
        """内建 hybrid vs 自研 RRF；顺带量 3k 块的 Linux 延迟。"""
        client2 = MilvusClient(DIR)
        name = "hybrid3k"
        client2.create_collection(name, schema=schema(extra=("kind",)), index_params=ip_for(client2, extra=("kind",)))
        vocab = ["锁等待", "事务隔离", "备份恢复", "分区表", "字符集", "OBProxy", "租户", "会话", "SQL 审计", "资源隔离", "副本", "合并"]
        body = " ".join(vocab)
        t0 = time.time()
        for s in range(0, 3000, 1000):
            client2.insert(
                name,
                [
                    {
                        "id": i,
                        "text": f"文档{i} {body} 第{i}节 说明",
                        "kind": "doc",
                        "vector": rng_vec(i),
                    }
                    for i in range(s, s + 1000)
                ],
            )
        insert_s = time.time() - t0
        settle(client2, name, 6.0, ("sparse", "vector"), warm_data=rng_vec(99999))

        q_text, q_vec = "锁等待的 SQL 怎么看", rng_vec(99999)
        lat = {"sparse": [], "dense": [], "hybrid_builtin": []}
        for _ in range(10):
            t = time.time(); client2.search(name, data=[q_text], anns_field="sparse", limit=50); lat["sparse"].append((time.time() - t) * 1000)
            t = time.time(); client2.search(name, data=[q_vec], anns_field="vector", limit=50); lat["dense"].append((time.time() - t) * 1000)
            t = time.time()
            reqs = [
                AnnSearchRequest(data=[q_text], anns_field="sparse", param={}, limit=50),
                AnnSearchRequest(data=[q_vec], anns_field="vector", param={}, limit=50),
            ]
            client2.hybrid_search(name, reqs=reqs, ranker=RRFRanker(60), limit=10)
            lat["hybrid_builtin"].append((time.time() - t) * 1000)
        reqs = [
            AnnSearchRequest(data=[q_text], anns_field="sparse", param={}, limit=50),
            AnnSearchRequest(data=[q_vec], anns_field="vector", param={}, limit=50),
        ]
        builtin = client2.hybrid_search(name, reqs=reqs, ranker=RRFRanker(60), limit=10)
        weighted = client2.hybrid_search(name, reqs=reqs, ranker=WeightedRanker(1.0, 1.0), limit=10)
        # 自研 RRF：两路各取 50，按 1/(k+rank) 合并
        sp = client2.search(name, data=[q_text], anns_field="sparse", limit=50)
        de = client2.search(name, data=[q_vec], anns_field="vector", limit=50)
        k = 60
        fused: dict[int, float] = {}
        for rank, h in enumerate(sp[0], 1):
            fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (k + rank)
        for rank, h in enumerate(de[0], 1):
            fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (k + rank)
        mine = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:10]]
        return {
            "insert_3k_s": round(insert_s, 1),
            "data_dir_mb": dirsize(DIR),
            "latency_ms_p50": {k2: round(statistics.median(v), 1) for k2, v in lat.items()},
            "latency_ms_max": {k2: round(max(v), 1) for k2, v in lat.items()},
            "builtin_top10": [h["id"] for h in builtin[0]],
            "weighted_top10": [h["id"] for h in weighted[0]],
            "manual_rrf_top10": mine,
            "builtin_scores": [round(float(h["distance"]), 6) for h in builtin[0][:3]],
        }

    def f7():
        """ob_meta 小集合共存 + query/filter；顺带确认「无向量字段的集合建不了」。"""
        client3 = MilvusClient(DIR)
        out = {}
        s0 = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        s0.add_field("key", DataType.VARCHAR, is_primary=True, max_length=64)
        s0.add_field("value", DataType.VARCHAR, max_length=512)
        try:
            client3.create_collection("ob_meta_novec", schema=s0, index_params=client3.prepare_index_params())
            out["no_vector_field"] = "accepted"
        except Exception as exc:  # noqa: BLE001
            out["no_vector_field"] = f"{type(exc).__name__}: {exc}"[:200]
        name = "ob_meta"
        s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        s.add_field("key", DataType.VARCHAR, is_primary=True, max_length=64)
        s.add_field("value", DataType.VARCHAR, max_length=512)
        s.add_field("num", DataType.DOUBLE)
        # 兜底：milvus-lite 要求集合必须带向量字段，元数据集合挂一个 2 维 dummy 向量。
        s.add_field("_dummy", DataType.FLOAT_VECTOR, dim=2)
        ip = client3.prepare_index_params()
        ip.add_index("_dummy", index_type="FLAT", metric_type="L2")
        client3.create_collection(name, schema=s, index_params=ip)
        client3.insert(
            name,
            [
                {"key": "schema_version", "value": "3", "num": 3.0, "_dummy": [0.0, 0.0]},
                {"key": "dims", "value": "1024", "num": 1024.0, "_dummy": [0.0, 0.0]},
            ],
        )
        time.sleep(2)
        rows = client3.query(name, filter='key in ["schema_version", "dims"]', output_fields=["key", "value", "num"])
        out.update({"rows": rows, "collections": sorted(client3.list_collections()), "data_dir_mb": dirsize(DIR)})
        return out

    def f8():
        """稠密索引选型：复用 f6 的 3k 集合，IVF_FLAT(nprobe) vs HNSW(ef)。"""
        import resource  # noqa: PLC0415

        c = MilvusClient(DIR)
        name = "hybrid3k"
        qv = rng_vec(99999)

        def bench(fn, reps=8, warm=2):
            for _ in range(warm):
                fn()
            ts = []
            for _ in range(reps):
                t = time.time()
                fn()
                ts.append((time.time() - t) * 1000)
            return {"p50": round(statistics.median(ts), 1), "min": round(min(ts), 1), "max": round(max(ts), 1)}

        c.load_collection(name)
        settle(c, name, 3.0, ("vector",), warm_data=qv)
        out = {"rss_mb_after_load": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
        out["desc_before"] = {
            k: v for k, v in c.describe_index(name, "vector").items() if k in ("index_type", "metric_type", "state", "total_rows")
        }
        out["ivf_flat_default"] = bench(lambda: c.search(name, data=[qv], anns_field="vector", limit=50))
        for nprobe in (1, 32):
            out[f"ivf_flat_nprobe{nprobe}"] = bench(
                lambda nprobe=nprobe: c.search(
                    name, data=[qv], anns_field="vector", limit=50, search_params={"params": {"nprobe": nprobe}}
                )
            )
        c.release_collection(name)
        time.sleep(1)
        c.drop_index(name, "vector")
        p = c.prepare_index_params()
        p.add_index("vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
        t = time.time()
        c.create_index(name, p)
        out["hnsw_build_s"] = round(time.time() - t, 1)
        for _ in range(300):
            d = c.describe_index(name, "vector")
            if d.get("pending_index_rows", 0) == 0:
                break
            time.sleep(1)
        out["hnsw_desc"] = {
            k: v for k, v in d.items() if k in ("index_type", "metric_type", "state", "total_rows")
        }
        c.load_collection(name)
        settle(c, name, 3.0, ("vector",), warm_data=qv)
        out["hnsw_default"] = bench(lambda: c.search(name, data=[qv], anns_field="vector", limit=50))
        for ef in (16, 200):
            out[f"hnsw_ef{ef}"] = bench(
                lambda ef=ef: c.search(name, data=[qv], anns_field="vector", limit=50, search_params={"params": {"ef": ef}})
            )
        out["dir_mb"] = dirsize(DIR)
        out["rss_mb_end"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        return out

    def f9():
        """可选（PROBE7_BIG=1）：25k 合成块暖机后的 sparse/dense/hybrid 延迟。

        probe5 在 macOS arm64 上量过同一个规模；这里是 Linux x86_64 / 2 vCPU 的对照，
        也是 M6 延迟门禁的唯一 Linux 参考点。正文长度按 1750 字（真实块上限 1800）构造。
        """
        import resource  # noqa: PLC0415

        shutil.rmtree(DIR_BIG, ignore_errors=True)
        c = MilvusClient(DIR_BIG)
        name = "big25k"
        c.create_collection(name, schema=schema(extra=("kind",)), index_params=ip_for(c, extra=("kind",)))
        vocab = [
            "锁等待", "事务隔离", "备份恢复", "分区表", "字符集", "OBProxy", "租户", "会话",
            "SQL 审计", "资源隔离", "副本", "合并", "系统变量", "执行计划", "慢查询", "连接池",
        ]
        body = "。".join(vocab * 12)[:1750]
        t0 = time.time()
        for s in range(0, 25000, 2000):
            c.insert(
                name,
                [{"id": i, "text": f"文档{i} {body}", "kind": "doc", "vector": rng_vec(i)} for i in range(s, s + 2000)],
            )
        insert_s = time.time() - t0
        qv = rng_vec(99999)
        settle(c, name, 10.0, ("sparse", "vector"), warm_data=qv)

        def bench(fn, reps=12, warm=3):
            for _ in range(warm):
                fn()
            ts = []
            for _ in range(reps):
                t = time.time()
                fn()
                ts.append((time.time() - t) * 1000)
            return {"p50": round(statistics.median(ts), 1), "min": round(min(ts), 1), "max": round(max(ts), 1)}

        reqs = [
            AnnSearchRequest(data=["锁等待的 SQL 怎么看"], anns_field="sparse", param={}, limit=50),
            AnnSearchRequest(data=[qv], anns_field="vector", param={}, limit=50),
        ]
        out = {
            "insert_25k_s": round(insert_s, 1),
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "sparse_l50": bench(lambda: c.search(name, data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=50)),
            "sparse_filter_doc_l50": bench(
                lambda: c.search(name, data=["锁等待的 SQL 怎么看"], anns_field="sparse", filter='kind == "doc"', limit=50)
            ),
            "dense_l50": bench(lambda: c.search(name, data=[qv], anns_field="vector", limit=50)),
            "hybrid_builtin_l10": bench(lambda: c.hybrid_search(name, reqs=reqs, ranker=RRFRanker(60), limit=10)),
        }
        out["data_dir_mb"] = dirsize(DIR_BIG)
        out["kb_per_chunk_1k"] = round(out["data_dir_mb"] * 1000 / 25000, 1)
        out["desc_dense"] = {
            k: v for k, v in c.describe_index(name, "vector").items() if k in ("index_type", "metric_type", "state", "total_rows")
        }
        return out

    sections = {
        "1": ("1_nullable_dense", f1),
        "2": ("2_varchar_max_length_unit", f2),
        "3": ("3_jieba_literal_queries", f3),
        "4": ("4_filter_exprs", f4),
        "5": ("5_multiprocess_lock", f5),
        "6": ("6_hybrid_and_latency", f6),
        "7": ("7_meta_collection", f7),
        "8": ("8_dense_index_choice", f8),
        "9": ("9_big25k_warm", f9),
    }
    # PROBE7_ONLY=9 只跑 25k 延迟段：干净进程 + 独立 data_dir（DIR_BIG），
    # 避免 f1–f8 的集合把 RSS 推到 1.4GB、把延迟推高一个数量级。
    only = [s.strip() for s in (os.environ.get("PROBE7_ONLY") or "").split(",") if s.strip()]
    if only:
        for key in only:
            if key not in sections:
                raise SystemExit(f"PROBE7_ONLY 只支持 {sorted(sections)}，收到 {key!r}")
            label, fn = sections[key]
            rec(label, fn)
    else:
        for key in ("1", "2", "3", "4", "5", "6", "7", "8"):
            label, fn = sections[key]
            rec(label, fn)
        if os.environ.get("PROBE7_BIG") == "1":
            rec("9_big25k_warm", f9)

    out_path = Path(__file__).with_name("probe7.out.json")
    merged = R
    if only and out_path.exists():
        # 只跑某几段时合并进已有结果，别把其余段的权威读数覆盖掉。
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev.update(R)
                merged = prev
        except (OSError, ValueError):
            merged = R
    payload = json.dumps(merged, ensure_ascii=False, indent=1, default=str)
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()