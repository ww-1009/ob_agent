import json, os, random, shutil, statistics, time
import numpy as np
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
R={}
def rec(n,fn):
    try: R[n]={"ok":True,"result":fn()}
    except Exception as e: R[n]={"ok":False,"error":f"{type(e).__name__}: {e}"[:300]}
DIR="/tmp/milvus_spike/data5.db"; shutil.rmtree(DIR,ignore_errors=True)
c=MilvusClient(DIR); J={"type":"jieba"}; DIM=1024
def dirsize(p):
    t=0
    for r,_d,fs in os.walk(p):
        for f in fs: t+=os.path.getsize(os.path.join(r,f))
    return round(t/1e6,1)
random.seed(5)
VOCAB=[f"术语{i}" for i in range(3000)] + ["锁等待","事务隔离","备份恢复","分区表","字符集","OBProxy","租户","会话","资源隔离","副本","合并","转储","日志流","选举","表组","SQL 审计"]
def real_text(i, nchar=1800):
    """非重复性真实形态正文：从大词表随机取词，避免病态 posting list"""
    ws=[VOCAB[random.randrange(len(VOCAB))] for _ in range(nchar//12)]
    return f"文档{i} " + " ".join(ws)

def build(N):
    n="corpus"
    s=MilvusClient.create_schema(auto_id=False,enable_dynamic_field=False)
    s.add_field("id",DataType.INT64,is_primary=True)
    s.add_field("text",DataType.VARCHAR,max_length=8000,enable_analyzer=True,analyzer_params=J)
    for f in ("mode","version"): s.add_field(f,DataType.VARCHAR,max_length=16)
    s.add_field("sparse",DataType.SPARSE_FLOAT_VECTOR)
    s.add_field("vector",DataType.FLOAT_VECTOR,dim=DIM)
    s.add_function(Function(name="bm25",function_type=FunctionType.BM25,input_field_names=["text"],output_field_names=["sparse"]))
    ip=c.prepare_index_params()
    ip.add_index("sparse",index_type="SPARSE_INVERTED_INDEX",metric_type="BM25")
    for f in ("mode","version"): ip.add_index(f,index_type="INVERTED")
    ip.add_index("vector",index_type="IVF_FLAT",metric_type="COSINE",params={"nlist":128})
    c.create_collection(n,schema=s,index_params=ip)
    rng=np.random.default_rng(5); t0=time.time()
    for st in range(0,N,2000):
        rows=[{"id":i,"text":real_text(i),"mode":random.choice(["MySQL","Oracle",""]),
               "version":random.choice(["4.2.5","4.3.5",""]),"vector":rng.random(DIM,dtype=np.float32).tolist()}
              for i in range(st,min(st+2000,N))]
        c.insert(n,rows)
    return n, round(time.time()-t0,1)

def f1():
    n,ins = build(25000)
    c.flush(n)
    # 等索引真正完成：轮询两个索引 + 打印状态
    hist=[]
    for k in range(60):
        ii=c.describe_index(n,"vector"); si=c.describe_index(n,"sparse")
        st={"t":k,"vec_pending":ii.get("pending_index_rows"),"vec_state":ii.get("state"),
            "sparse_pending":si.get("pending_index_rows"),"sparse_state":si.get("state"),
            "row_count":c.get_collection_stats(n).get("row_count")}
        hist.append(st)
        if ii.get("state")=="Finished" and si.get("state")=="Finished" and (ii.get("pending_index_rows") or 0)==0: break
        time.sleep(2)
    load=c.get_load_state(n)
    return {"insert_25k_s":ins,"index_wait_iters":k,"last_status":hist[-1],
            "load_state":str(load)[:160],"data_dir_mb":dirsize(DIR),
            "bytes_per_chunk":round(dirsize(DIR)*1e6/25000,1)}

def f2():
    n="corpus"; time.sleep(20)   # 额外静置，确保 compaction 完成
    out={}
    qs=["锁等待的 SQL 怎么看","事务隔离级别实现","ORA-00942 报错怎么处理"]
    for q in qs:
        ts=[]
        for k in range(23):
            t1=time.time(); c.search(n,data=[q],anns_field="sparse",limit=50); ms=(time.time()-t1)*1000
            if k>=3: ts.append(ms)
        out[f"sparse:{q}"]={"p50":round(statistics.median(ts),1),"max":round(max(ts),1)}
    rng=np.random.default_rng(9)
    for lim in (10,50):
        ts=[]
        for k in range(23):
            qv=rng.random(DIM,dtype=np.float32).tolist()
            t1=time.time(); c.search(n,data=[qv],anns_field="vector",limit=lim); ms=(time.time()-t1)*1000
            if k>=3: ts.append(ms)
        out[f"dense:limit={lim}"]={"p50":round(statistics.median(ts),1),"max":round(max(ts),1)}
    return {"steady_state_latency_ms":out,"data_dir_mb":dirsize(DIR)}

def f3():
    n="corpus"
    reqs=[AnnSearchRequest(data=["锁等待的 SQL 怎么看"],anns_field="sparse",param={},limit=50),
          AnnSearchRequest(data=[np.random.default_rng(1).random(DIM,dtype=np.float32).tolist()],anns_field="vector",param={},limit=50)]
    ts=[]
    for k in range(13):
        t1=time.time(); c.hybrid_search(n,reqs=reqs,ranker=RRFRanker(60),limit=10); ms=(time.time()-t1)*1000
        if k>=3: ts.append(ms)
    return {"builtin_hybrid_rrf_p50_ms":round(statistics.median(ts),1),"max_ms":round(max(ts),1)}

for i,(n,fn) in enumerate([("1_build_25k_real",f1),("2_steady_latency",f2),("3_hybrid_latency",f3)],1):
    rec(n,fn)
print(json.dumps(R,ensure_ascii=False,indent=1,default=str))
