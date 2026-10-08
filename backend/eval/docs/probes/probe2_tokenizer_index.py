import json, os, shutil, statistics, time, traceback
import numpy as np
from pymilvus import (MilvusClient, DataType, Function, FunctionType,
                      AnnSearchRequest, RRFRanker)
R = {}
def rec(name, fn):
    try: R[name] = {"ok": True, "result": fn()}
    except Exception as e:
        R[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}

DIR = "/tmp/milvus_spike/data2.db"
shutil.rmtree(DIR, ignore_errors=True)
c = MilvusClient(DIR)
tok = lambda o: o if isinstance(o, str) else str(o)[:400]
DIM = 1024

# ---------- 1. 内建 BM25：enable_analyzer=True ----------
def f1():
    name = "bm25_ok"
    sch = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("id", DataType.INT64, is_primary=True)
    sch.add_field("text", DataType.VARCHAR, max_length=4000,
                  enable_analyzer=True, analyzer_params={"type": "chinese"})
    sch.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    sch.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                              input_field_names=["text"], output_field_names=["sparse"]))
    ip = c.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    c.create_collection(name, schema=sch, index_params=ip)
    c.insert(name, [
        {"id": 1, "text": "如何查看锁等待的 SQL 语句"},
        {"id": 2, "text": "Oracle 模式的事务隔离级别是怎么实现的"},
        {"id": 3, "text": "ORA-00942 表或视图不存在 错误码处理"},
        {"id": 4, "text": "OceanBase 4.2.5 版本升级注意事项"},
        {"id": 5, "text": "备份恢复总览"},
    ])
    time.sleep(0.5)
    out = {}
    for q in ["锁等待的 SQL 怎么看", "ORA-00942", "4.2.5 版本升级", "事务隔离级别"]:
        r = c.search(name, data=[q], anns_field="sparse", limit=3, output_fields=["text"])
        out[q] = [(h["id"], round(h["distance"], 4)) for h in r[0]]
    return out

# ---------- 2. 分析器实际切出的 token（决定 literal 门禁风险）----------
def f2():
    out = {}
    for label, params in {"chinese": {"type": "chinese"}, "standard": {"type": "standard"}}.items():
        try:
            r = c.run_analyzer("ORA-00942 表或视图不存在 4.2.5 锁等待 SQL", analyzer_params=params)
            toks = r.get("tokens", r) if isinstance(r, dict) else r
            out[label] = [t.get("token") if isinstance(t, dict) else str(t) for t in toks]
        except Exception as e:
            out[label] = f"ERR {type(e).__name__}: {e}"[:150]
    return out

# ---------- 3. BM25 规模与延迟（24k 文本）----------
def f3():
    name = "bm25_scale"
    sch = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("id", DataType.INT64, is_primary=True)
    sch.add_field("text", DataType.VARCHAR, max_length=4000,
                  enable_analyzer=True, analyzer_params={"type": "chinese"})
    sch.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    sch.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                              input_field_names=["text"], output_field_names=["sparse"]))
    ip = c.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    c.create_collection(name, schema=sch, index_params=ip)
    vocab = ["锁等待", "事务隔离", "备份恢复", "分区表", "字符集", "OBProxy", "租户", "会话",
             "SQL 审计", "资源隔离", "副本", "合并", "转储", "日志流", "选举", "表组"]
    N, B = 24000, 3000
    t0 = time.time()
    for s in range(0, N, B):
        rows = [{"id": i, "text": f"文档{i} " + " ".join(vocab[(i+j) % len(vocab)] for j in range(6))}
                for i in range(s, min(s+B, N))]
        c.insert(name, rows)
    ins = time.time() - t0
    lat = {}
    for q, lim in [("锁等待的 SQL 怎么看", 50), ("ORA-00942", 50), ("备份恢复", 10)]:
        ts = []
        for _ in range(15):
            t1 = time.time(); c.search(name, data=[q], anns_field="sparse", limit=lim); ts.append((time.time()-t1)*1000)
        lat[f"{q}|limit={lim}"] = {"p50_ms": round(statistics.median(ts), 1), "max_ms": round(max(ts), 1)}
    return {"insert_24k_s": round(ins, 1), "latency": lat,
            "num_entities": c.get_collection_stats(name).get("row_count")}

# ---------- 4. IDF 稳定性：写入新数据后同一 query 分数是否漂移 ----------
def f4():
    name = "bm25_scale"
    b = {h["id"]: round(h["distance"], 6) for h in
         c.search(name, data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=5)[0]}
    c.insert(name, [{"id": 90000 + i, "text": f"锁等待 排查 手册 第{i}版 行锁 表锁 全局锁"} for i in range(300)])
    time.sleep(1.0)
    a = {h["id"]: round(h["distance"], 6) for h in
         c.search(name, data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=5)[0]}
    common = sorted(set(b) & set(a))
    return {"before": {k: b[k] for k in common}, "after": {k: a[k] for k in common},
            "drifted": {k: [b[k], a[k]] for k in common if abs(b[k]-a[k]) > 1e-6}}

# ---------- 5. BM25 + 标量过滤 组合 ----------
def f5():
    name = "bm25_filter"
    sch = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("id", DataType.INT64, is_primary=True)
    sch.add_field("text", DataType.VARCHAR, max_length=2000, enable_analyzer=True,
                  analyzer_params={"type": "chinese"})
    sch.add_field("mode", DataType.VARCHAR, max_length=16)
    sch.add_field("version", DataType.VARCHAR, max_length=16)
    sch.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    sch.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                              input_field_names=["text"], output_field_names=["sparse"]))
    ip = c.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for f in ("mode", "version"):
        ip.add_index(f, index_type="INVERTED")
    c.create_collection(name, schema=sch, index_params=ip)
    c.insert(name, [{"id": 1, "text": "锁等待 SQL 查询", "mode": "MySQL", "version": "4.2.5"},
                    {"id": 2, "text": "锁等待 SQL 查询", "mode": "Oracle", "version": "4.3.5"},
                    {"id": 3, "text": "锁等待 SQL 查询", "mode": "", "version": ""}])
    time.sleep(0.5)
    f = '(mode == "MySQL" or mode == "") and (version == "" or version like "4.2.5%")'
    r = c.search(name, data=["锁等待"], anns_field="sparse", filter=f, limit=10, output_fields=["mode","version"])
    return {"filtered_ids": sorted(h["id"] for h in r[0]), "filter": f}

# ---------- 6. 稠密索引类型对比（24k x 1024, limit=50）----------
def f6():
    out = {}
    rng = np.random.default_rng(7)
    vecs = rng.random((24000, DIM), dtype=np.float32)
    for itype in ["FLAT", "HNSW", "AUTOINDEX", "IVF_FLAT"]:
        name = f"dense_{itype.lower()}"
        try:
            sch = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            sch.add_field("id", DataType.INT64, is_primary=True)
            sch.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
            ip = c.prepare_index_params()
            kw = {"index_type": itype, "metric_type": "COSINE"}
            if itype == "HNSW": kw["params"] = {"M": 16, "efConstruction": 200}
            if itype == "IVF_FLAT": kw["params"] = {"nlist": 128}
            ip.add_index("vector", **kw)
            c.create_collection(name, schema=sch, index_params=ip)
            for s in range(0, 24000, 4000):
                c.insert(name, [{"id": i, "vector": vecs[i].tolist()} for i in range(s, s+4000)])
            time.sleep(0.5)
            q = vecs[0].tolist()
            ts = []
            for _ in range(15):
                t1 = time.time(); c.search(name, data=[q], limit=50); ts.append((time.time()-t1)*1000)
            info = c.describe_index(name, "vector")
            out[itype] = {"p50_ms": round(statistics.median(ts), 1), "max_ms": round(max(ts), 1),
                          "index_type_reported": info.get("index_type")}
            c.drop_collection(name)
        except Exception as e:
            out[itype] = f"ERR {type(e).__name__}: {e}"[:200]
    return out

# ---------- 7. 内建 hybrid search（RRFRanker）是否可用 ----------
def f7():
    name = "hybrid_test"
    sch = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("id", DataType.INT64, is_primary=True)
    sch.add_field("text", DataType.VARCHAR, max_length=2000, enable_analyzer=True,
                  analyzer_params={"type": "chinese"})
    sch.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    sch.add_field("vector", DataType.FLOAT_VECTOR, dim=8)
    sch.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                              input_field_names=["text"], output_field_names=["sparse"]))
    ip = c.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    ip.add_index("vector", index_type="FLAT", metric_type="COSINE")
    c.create_collection(name, schema=sch, index_params=ip)
    c.insert(name, [{"id": 1, "text": "锁等待 SQL", "vector": [1.0]+[0.0]*7},
                    {"id": 2, "text": "备份恢复", "vector": [0.0, 1.0]+[0.0]*6}])
    time.sleep(0.5)
    reqs = [AnnSearchRequest(data=["锁等待"], anns_field="sparse", param={}, limit=10),
            AnnSearchRequest(data=[[1.0]+[0.0]*7], anns_field="vector", param={}, limit=10)]
    r = c.hybrid_search(name, reqs=reqs, ranker=RRFRanker(60), limit=5, output_fields=["text"])
    return {"hits": [(h["id"], h["entity"]["text"]) for h in r[0]]}

for i, (n, fn) in enumerate([("1_bm25_enable_analyzer", f1), ("2_analyzer_tokens", f2),
                             ("3_bm25_scale_latency", f3), ("4_bm25_idf_drift", f4),
                             ("5_bm25_scalar_filter", f5), ("6_dense_index_types", f6),
                             ("7_builtin_hybrid_rrf", f7)], 1):
    rec(n, fn)
print(json.dumps(R, ensure_ascii=False, indent=1, default=tok))
