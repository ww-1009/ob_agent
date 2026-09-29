import json, shutil, statistics, time, traceback
import numpy as np
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
R = {}
def rec(n, fn):
    try: R[n] = {"ok": True, "result": fn()}
    except Exception as e: R[n] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:260]}

DIR = "/tmp/milvus_spike/data3.db"; shutil.rmtree(DIR, ignore_errors=True)
c = MilvusClient(DIR); J = {"type": "jieba"}; DIM = 1024

def bm25_schema(text_max=4000):
    sch = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("id", DataType.INT64, is_primary=True)
    sch.add_field("text", DataType.VARCHAR, max_length=text_max, enable_analyzer=True, analyzer_params=J)
    sch.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    sch.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                              input_field_names=["text"], output_field_names=["sparse"]))
    return sch
def bm25_index():
    ip = c.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    return ip
def wait_ready(name, expect):
    for _ in range(60):
        st = c.get_collection_stats(name).get("row_count", 0)
        idx = c.describe_index(name, "sparse") if "sparse" in str(c.list_indexes(name)) else {}
        if st >= expect and (not idx or idx.get("pending_index_rows", 0) == 0): return st
        time.sleep(1)
    return c.get_collection_stats(name).get("row_count", 0)

# 1) 内建 BM25 基本可用性 + literal 行为
def f1():
    n = "bm25_lit"; c.create_collection(n, schema=bm25_schema(2000), index_params=bm25_index())
    c.insert(n, [{"id": 1, "text": "如何查看锁等待的 SQL 语句"}, {"id": 2, "text": "Oracle 模式事务隔离级别实现"},
                 {"id": 3, "text": "ORA-00942 表或视图不存在 错误码"}, {"id": 4, "text": "OceanBase 4.2.5 版本升级注意事项"},
                 {"id": 5, "text": "备份恢复总览"}])
    wait_ready(n, 5); out = {}
    for q in ["锁等待的 SQL 怎么看", "ORA-00942", "00942", "4.2.5 版本升级", "事务隔离级别"]:
        r = c.search(n, data=[q], anns_field="sparse", limit=3, output_fields=["text"])
        out[q] = [(h["id"], round(h["distance"], 3)) for h in r[0]]
    return out

# 2) BM25 规模 + 延迟（等索引就绪）
def f2():
    n = "bm25_scale"; c.create_collection(n, schema=bm25_schema(4000), index_params=bm25_index())
    vocab = ["锁等待","事务隔离","备份恢复","分区表","字符集","OBProxy","租户","会话","SQL 审计","资源隔离","副本","合并"]
    t0 = time.time()
    for s in range(0, 24000, 3000):
        c.insert(n, [{"id": i, "text": f"文档{i} " + " ".join(vocab[(i+j)%12] for j in range(6))} for i in range(s, s+3000)])
    ins = time.time()-t0
    wait_ready(n, 24000); time.sleep(2)
    lat = {}
    for q, lim in [("锁等待的 SQL 怎么看", 50), ("备份恢复", 10)]:
        ts = []
        for _ in range(15):
            t1 = time.time(); c.search(n, data=[q], anns_field="sparse", limit=lim); ts.append((time.time()-t1)*1000)
        lat[f"{q}|limit={lim}"] = {"p50": round(statistics.median(ts),1), "max": round(max(ts),1)}
    return {"insert_24k_s": round(ins,1), "latency_ms": lat, "rows": c.get_collection_stats(n).get("row_count")}

# 3) IDF 段内漂移
def f3():
    n = "bm25_scale"
    b = {h["id"]: round(h["distance"],6) for h in c.search(n, data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=5)[0]}
    c.insert(n, [{"id": 90000+i, "text": f"锁等待 排查 手册 第{i}版 行锁 表锁 全局锁"} for i in range(300)])
    time.sleep(3)
    a = {h["id"]: round(h["distance"],6) for h in c.search(n, data=["锁等待的 SQL 怎么看"], anns_field="sparse", limit=5)[0]}
    common = sorted(set(b)&set(a))
    return {"pairs": {k: [b[k], a[k]] for k in common}, "drifted": [k for k in common if abs(b[k]-a[k])>1e-6]}

# 4) BM25 + 标量过滤
def f4():
    n = "bm25_filter"
    sch = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("id", DataType.INT64, is_primary=True)
    sch.add_field("text", DataType.VARCHAR, max_length=2000, enable_analyzer=True, analyzer_params=J)
    for f in ("mode","version"): sch.add_field(f, DataType.VARCHAR, max_length=16)
    sch.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    sch.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                              input_field_names=["text"], output_field_names=["sparse"]))
    ip = c.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for f in ("mode","version"): ip.add_index(f, index_type="INVERTED")
    c.create_collection(n, schema=sch, index_params=ip)
    c.insert(n, [{"id":1,"text":"锁等待 SQL 查询","mode":"MySQL","version":"4.2.5"},
                 {"id":2,"text":"锁等待 SQL 查询","mode":"Oracle","version":"4.3.5"},
                 {"id":3,"text":"锁等待 SQL 查询","mode":"","version":""}])
    wait_ready(n,3); time.sleep(1)
    f='(mode == "MySQL" or mode == "") and (version == "" or version like "4.2.5%")'
    r=c.search(n,data=["锁等待"],anns_field="sparse",filter=f,limit=10,output_fields=["mode","version"])
    return {"ids": sorted(h["id"] for h in r[0]), "filter": f}

# 5) 内建 hybrid RRF
def f5():
    n="hybrid"; sch=MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("id", DataType.INT64, is_primary=True)
    sch.add_field("text", DataType.VARCHAR, max_length=2000, enable_analyzer=True, analyzer_params=J)
    sch.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    sch.add_field("vector", DataType.FLOAT_VECTOR, dim=8)
    sch.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                              input_field_names=["text"], output_field_names=["sparse"]))
    ip=c.prepare_index_params()
    ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    ip.add_index("vector", index_type="FLAT", metric_type="COSINE")
    c.create_collection(n, schema=sch, index_params=ip)
    c.insert(n,[{"id":1,"text":"锁等待 SQL","vector":[1.0]+[0.0]*7},{"id":2,"text":"备份恢复","vector":[0.0,1.0]+[0.0]*6}])
    wait_ready(n,2); time.sleep(1)
    reqs=[AnnSearchRequest(data=["锁等待"],anns_field="sparse",param={},limit=10),
          AnnSearchRequest(data=[[1.0]+[0.0]*7],anns_field="vector",param={},limit=10)]
    r=c.hybrid_search(n,reqs=reqs,ranker=RRFRanker(60),limit=5,output_fields=["text"])
    return {"hits":[(h["id"],h["entity"]["text"]) for h in r[0]]}

# 6) 干净重测稠密延迟（等索引就绪）
def f6():
    out={}; rng=np.random.default_rng(11); vecs=rng.random((24000,DIM),dtype=np.float32)
    for itype,kw in [("FLAT",{}),("IVF_FLAT",{"params":{"nlist":128}}),("HNSW",{"params":{"M":16,"efConstruction":200}})]:
        n=f"d_{itype.lower()}"
        sch=MilvusClient.create_schema(auto_id=False,enable_dynamic_field=False)
        sch.add_field("id",DataType.INT64,is_primary=True); sch.add_field("vector",DataType.FLOAT_VECTOR,dim=DIM)
        ip=c.prepare_index_params(); ip.add_index("vector",index_type=itype,metric_type="COSINE",**kw)
        c.create_collection(n,schema=sch,index_params=ip)
        for s in range(0,24000,4000):
            c.insert(n,[{"id":i,"vector":vecs[i].tolist()} for i in range(s,s+4000)])
        for _ in range(30):
            if c.describe_index(n,"vector").get("pending_index_rows",0)==0: break
            time.sleep(1)
        time.sleep(2)
        q=vecs[0].tolist(); r={}
        for lim in (10,50):
            ts=[]
            for _ in range(20):
                t1=time.time(); c.search(n,data=[q],limit=lim); ts.append((time.time()-t1)*1000)
            r[f"limit={lim}"]={"p50":round(statistics.median(ts),1),"max":round(max(ts),1)}
        out[itype]=r; c.drop_collection(n)
    return out

# 7) upsert / delete（增量缓存可行性）
def f7():
    n="up_test"; c.create_collection(n,schema=bm25_schema(500),index_params=bm25_index())
    c.insert(n,[{"id":1,"text":"第一版 内容 锁等待"},{"id":2,"text":"保留 文档 备份"}])
    wait_ready(n,2); time.sleep(1)
    before={h["id"]:h["entity"]["text"] for h in c.search(n,data=["文档"],anns_field="sparse",limit=5,output_fields=["text"])[0]}
    c.upsert(n,[{"id":1,"text":"第二版 内容 事务隔离"}]); time.sleep(2)
    after={h["id"]:h["entity"]["text"] for h in c.search(n,data=["事务隔离"],anns_field="sparse",limit=5,output_fields=["text"])[0]}
    c.delete(n,ids=[2]); time.sleep(1)
    rows=c.get_collection_stats(n).get("row_count")
    return {"before_search_doc":before,"after_upsert_search":after,"rows_after_delete":rows}

for i,(n,fn) in enumerate([("1_bm25_literal",f1),("2_bm25_scale_latency",f2),("3_idf_drift",f3),
                           ("4_bm25_filter",f4),("5_builtin_hybrid",f5),("6_dense_latency_clean",f6),
                           ("7_upsert_delete",f7)],1):
    rec(n,fn)
print(json.dumps(R,ensure_ascii=False,indent=1,default=str))
