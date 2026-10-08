import json, os, shutil, subprocess, sys, time, traceback
R = {}
def t(name):
    def deco(fn):
        try:
            r = fn()
            R[name] = {"ok": True, "result": r}
        except Exception as e:
            R[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400],
                       "tb": traceback.format_exc().splitlines()[-3:]}
        return fn
    return deco

import pymilvus
from pymilvus import (MilvusClient, DataType, Function, FunctionType,
                      AnnSearchRequest, RRFRanker, WeightedRanker)
R["versions"] = {"pymilvus": pymilvus.__version__}
try:
    import milvus_lite; R["versions"]["milvus_lite"] = getattr(milvus_lite, "__version__", "?")
except Exception as e:
    R["versions"]["milvus_lite"] = f"import failed: {e}"

DIR = "/tmp/milvus_spike/data.db"
shutil.rmtree(DIR, ignore_errors=True)

@t("1_client_open")
def _():
    c = MilvusClient(DIR)
    return {"uri_ok": True}

@t("2_dense_flat")
def _():
    c = MilvusClient(DIR); name = "dense_flat"
    c.create_collection(name, dimension=1024, metric_type="COSINE", auto_id=False)
    # 24k x 1024 随机向量，批量插入
    import random
    N, B = 24000, 2000
    t0 = time.time()
    for s in range(0, N, B):
        rows = [{"id": i, "vector": [random.random() for _ in range(1024)]} for i in range(s, min(s+B, N))]
        c.insert(name, rows)
    ins = time.time() - t0
    lat = []
    q = [random.random() for _ in range(1024)]
    for _i in range(20):
        t1 = time.time(); c.search(name, data=[q], limit=50); lat.append((time.time()-t1)*1000)
    lat.sort()
    return {"insert_24k_s": round(ins, 1), "search_p50_ms": round(lat[len(lat)//2], 1),
            "search_max_ms": round(lat[-1], 1), "index": str(c.describe_index(name, "vector"))[:300]}

@t("3_sparse_user_vectors")
def _():
    """方案(ii)：自建稀疏向量（自己分词+全局IDF权重）注入 SPARSE_FLOAT_VECTOR"""
    c = MilvusClient(DIR); name = "sparse_user"
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("text", DataType.VARCHAR, max_length=512)
    idx = c.prepare_index_params()
    idx.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    c.create_collection(name, schema=schema, index_params=idx)
    # 模拟自写单双字分词后的稀疏向量
    docs = [{"id": 1, "text": "查询行锁", "sparse": {101: 1.5, 202: 0.8, 303: 2.1}},
            {"id": 2, "text": "事务隔离级别", "sparse": {404: 1.1, 505: 0.9}},
            {"id": 3, "text": "锁等待 SQL", "sparse": {101: 0.7, 606: 1.3}}]
    c.insert(name, docs)
    res = c.search(name, data=[{101: 1.0, 303: 1.0}], anns_field="sparse", limit=3,
                   output_fields=["text"])
    return {"hits": [[(h["id"], round(h["distance"], 3), h["entity"]["text"]) for h in r] for r in res]}

@t("4_bm25_function")
def _():
    """方案(i)：Milvus 内建 BM25 Function（服务端分析器）"""
    c = MilvusClient(DIR); name = "bm25_func"
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=512,
                     analyzer_params={"type": "chinese"})
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                                input_field_names=["text"], output_field_names=["sparse"]))
    idx = c.prepare_index_params()
    idx.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    c.create_collection(name, schema=schema, index_params=idx)
    c.insert(name, [{"id": 1, "text": "如何查看锁等待的 SQL"},
                    {"id": 2, "text": "事务隔离级别实现"},
                    {"id": 3, "text": "ORA-00942 表或视图不存在"}])
    return {"insert_ok": True, "has_function": "ok"}

@t("5_bm25_search")
def _():
    c = MilvusClient(DIR)
    res = c.search("bm25_func", data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=3,
                   output_fields=["text"])
    return {"hits": [[(h["id"], round(h["distance"], 3), h["entity"]["text"]) for h in r] for r in res]}

@t("6_bm25_idf_stability")
def _():
    """加数据后同一 query 的分数是否漂移（段内 IDF 之疑）"""
    c = MilvusClient(DIR)
    before = c.search("bm25_func", data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=3)[0]
    b = {h["id"]: round(h["distance"], 6) for h in before}
    c.insert("bm25_func", [{"id": 10 + i, "text": f"锁等待排查方法之{i} 全局锁 表锁 行锁"} for i in range(200)])
    time.sleep(1)
    after = c.search("bm25_func", data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=3)[0]
    a = {h["id"]: round(h["distance"], 6) for h in after}
    common = {k: (b[k], a[k]) for k in set(b) & set(a)}
    return {"before": b, "after": a, "changed": {k: v for k, v in common.items() if v[0] != v[1]}}

@t("7_scalar_filter")
def _():
    c = MilvusClient(DIR); name = "filter_test"
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("mode", DataType.VARCHAR, max_length=32)
    schema.add_field("version", DataType.VARCHAR, max_length=32)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=8)
    idx = c.prepare_index_params(); idx.add_index("vector", index_type="FLAT", metric_type="COSINE")
    c.create_collection(name, schema=schema, index_params=idx)
    c.insert(name, [{"id": 1, "mode": "MySQL", "version": "4.2.5", "vector": [0.1]*8},
                    {"id": 2, "mode": "", "version": "", "vector": [0.2]*8},
                    {"id": 3, "mode": "Oracle", "version": "4.3.5", "vector": [0.3]*8}])
    out = {}
    for label, f in {"mode_eq_or_empty": 'mode == "MySQL" or mode == ""',
                     "version_prefix": 'version like "4.2.5%"',
                     "combined": '(mode == "MySQL" or mode == "") and (version == "" or version like "4.2.5%")'}.items():
        try:
            r = c.search(name, data=[[0.1]*8], filter=f, limit=10, output_fields=["mode","version"])
            out[label] = sorted(h["id"] for h in r[0])
        except Exception as e:
            out[label] = f"ERR {type(e).__name__}: {e}"[:200]
    return out

@t("8_data_dir_layout")
def _():
    files, total = [], 0
    for root, _d, fs in os.walk(DIR):
        for f in fs:
            fp = os.path.join(root, f); sz = os.path.getsize(fp)
            total += sz; files.append((os.path.relpath(fp, DIR), sz))
    files.sort(key=lambda x: -x[1])
    return {"is_dir": os.path.isdir(DIR), "file_count": len(files), "total_mb": round(total/1e6, 1),
            "top": files[:8]}

@t("9_multiprocess_lock")
def _():
    """第二个进程开同一 data_dir 是否失败（--workers 1 约束）"""
    code = "import sys; from pymilvus import MilvusClient\ntry:\n MilvusClient(sys.argv[1]); print('OPENED')\nexcept Exception as e:\n print('BLOCKED', type(e).__name__, str(e)[:150])\n"
    r = subprocess.run([sys.executable, "-c", code, DIR], capture_output=True, text=True, timeout=120)
    return {"rc": r.returncode, "out": (r.stdout + r.stderr).strip()[:300]}

print(json.dumps(R, ensure_ascii=False, indent=1))
