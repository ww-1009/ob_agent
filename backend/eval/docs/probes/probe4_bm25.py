import json, os, shutil, statistics, time
import numpy as np
from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
R={}
def rec(n,fn):
    try: R[n]={"ok":True,"result":fn()}
    except Exception as e: R[n]={"ok":False,"error":f"{type(e).__name__}: {e}"[:260]}
DIR="/tmp/milvus_spike/data4.db"; shutil.rmtree(DIR,ignore_errors=True)
c=MilvusClient(DIR); J={"type":"jieba"}; DIM=1024
def sch_bm25(tmax=8000, extra_scalars=True, dense=False):
    s=MilvusClient.create_schema(auto_id=False,enable_dynamic_field=False)
    s.add_field("id",DataType.INT64,is_primary=True)
    s.add_field("text",DataType.VARCHAR,max_length=tmax,enable_analyzer=True,analyzer_params=J)
    s.add_field("sparse",DataType.SPARSE_FLOAT_VECTOR)
    if extra_scalars:
        for f in ("mode","version"): s.add_field(f,DataType.VARCHAR,max_length=16)
    if dense: s.add_field("vector",DataType.FLOAT_VECTOR,dim=DIM)
    s.add_function(Function(name="bm25",function_type=FunctionType.BM25,input_field_names=["text"],output_field_names=["sparse"]))
    return s
def ip_bm25(extra=(), dense=False, dense_type="IVF_FLAT"):
    ip=c.prepare_index_params()
    ip.add_index("sparse",index_type="SPARSE_INVERTED_INDEX",metric_type="BM25")
    for f in extra: ip.add_index(f,index_type="INVERTED")
    if dense: ip.add_index("vector",index_type=dense_type,metric_type="COSINE",params={"nlist":128})
    return ip
def ready(n,expect,field="sparse"):
    for _ in range(90):
        if c.describe_index(n,field).get("pending_index_rows",0)==0 and c.get_collection_stats(n).get("row_count",0)>=expect: return True
        time.sleep(1)
    return False
def dirsize(p):
    tot=0
    for r,_d,fs in os.walk(p):
        for f in fs: tot+=os.path.getsize(os.path.join(r,f))
    return round(tot/1e6,1)

def f1():
    n="lit"; c.create_collection(n,schema=sch_bm25(2000,extra_scalars=False),index_params=ip_bm25())
    c.insert(n,[{"id":1,"text":"如何查看锁等待的 SQL 语句"},{"id":2,"text":"Oracle 模式事务隔离级别实现"},
                {"id":3,"text":"ORA-00942 表或视图不存在 错误码"},{"id":4,"text":"OceanBase 4.2.5 版本升级注意事项"},
                {"id":5,"text":"备份恢复总览"}])
    ready(n,5); out={}
    for q in ["锁等待的 SQL 怎么看","ORA-00942","00942","4.2.5 版本升级","事务隔离级别","备份"]:
        r=c.search(n,data=[q],anns_field="sparse",limit=3,output_fields=["text"])
        out[q]=[(h["id"],round(h["distance"],3)) for h in r[0]]
    return out

def f2():
    n="scale"; c.create_collection(n,schema=sch_bm25(),index_params=ip_bm25(extra=("mode","version")))
    vocab=["锁等待","事务隔离","备份恢复","分区表","字符集","OBProxy","租户","会话","SQL 审计","资源隔离","副本","合并"]
    t0=time.time()
    for s in range(0,24000,3000):
        c.insert(n,[{"id":i,"text":f"文档{i} "+" ".join(vocab[(i+j)%12] for j in range(6)),"mode":"MySQL","version":"4.2.5"} for i in range(s,s+3000)])
    ins=time.time()-t0
    ok=ready(n,24000); time.sleep(2)
    lat={}
    for q,lim in [("锁等待的 SQL 怎么看",50),("备份恢复",10)]:
        ts=[]
        for _ in range(15):
            t1=time.time(); c.search(n,data=[q],anns_field="sparse",limit=lim); ts.append((time.time()-t1)*1000)
        lat[f"{q}|limit={lim}"]={"p50":round(statistics.median(ts),1),"max":round(max(ts),1)}
    return {"insert_24k_s":round(ins,1),"index_ready":ok,"rows":c.get_collection_stats(n).get("row_count"),
            "latency_ms":lat,"data_dir_mb":dirsize(DIR)}

def f3():
    n="scale"
    b={h["id"]:round(h["distance"],6) for h in c.search(n,data=["锁等待的 SQL 怎么看"],anns_field="sparse",limit=5)[0]}
    c.insert(n,[{"id":90000+i,"text":f"锁等待 排查 手册 第{i}版 行锁 表锁 全局锁","mode":"Oracle","version":"4.3.5"} for i in range(300)])
    time.sleep(4)
    a={h["id"]:round(h["distance"],6) for h in c.search(n,data=["锁等待的 SQL 怎么看"],anns_field="sparse",limit=5)[0]}
    common=sorted(set(b)&set(a))
    return {"pairs":{k:[b[k],a[k]] for k in common},"drifted":[k for k in common if abs(b[k]-a[k])>1e-6]}

def f4():
    n="scale"; f='(mode == "MySQL" or mode == "") and (version == "" or version like "4.2.5%")'
    r=c.search(n,data=["锁等待"],anns_field="sparse",filter=f,limit=10,output_fields=["mode","version"])
    n2=c.search(n,data=["锁等待"],anns_field="sparse",limit=10,output_fields=["mode","version"])
    return {"filtered":[(h["id"],h["entity"]["mode"]) for h in r[0]][:4],
            "unfiltered_modes":sorted({h["entity"]["mode"] for h in n2[0]}),"filter":f}

def f5():
    n="hyb"; c.create_collection(n,schema=sch_bm25(2000,extra_scalars=False,dense=True),index_params=ip_bm25(dense=True,dense_type="FLAT"))
    c.insert(n,[{"id":1,"text":"锁等待 SQL","vector":[1.0]+[0.0]*1023},{"id":2,"text":"备份恢复","vector":[0.0,1.0]+[0.0]*1022}])
    ready(n,2); time.sleep(1)
    reqs=[AnnSearchRequest(data=["锁等待"],anns_field="sparse",param={},limit=10),
          AnnSearchRequest(data=[[1.0]+[0.0]*1023],anns_field="vector",param={},limit=10)]
    r=c.hybrid_search(n,reqs=reqs,ranker=RRFRanker(60),limit=5,output_fields=["text"])
    return {"hits":[(h["id"],h["entity"]["text"]) for h in r[0]]}

def f6():
    n="up"; c.create_collection(n,schema=sch_bm25(500,extra_scalars=False),index_params=ip_bm25())
    c.insert(n,[{"id":1,"text":"第一版 内容 锁等待"},{"id":2,"text":"保留 文档 备份"}])
    ready(n,2); time.sleep(1)
    b={h["id"]:h["entity"]["text"] for h in c.search(n,data=["文档"],anns_field="sparse",limit=5,output_fields=["text"])[0]}
    c.upsert(n,[{"id":1,"text":"第二版 内容 事务隔离"}]); time.sleep(2)
    a={h["id"]:h["entity"]["text"] for h in c.search(n,data=["事务隔离"],anns_field="sparse",limit=5,output_fields=["text"])[0]}
    c.delete(n,ids=[2]); time.sleep(1)
    return {"before":b,"after_upsert":a,"rows_after_delete":c.get_collection_stats(n).get("row_count")}

def f7():
    """真实形态：1800 字中文正文 + sparse + dense，5000 块，量容量与两端延迟"""
    n="realistic"; c.create_collection(n,schema=sch_bm25(8000,dense=True),index_params=ip_bm25(extra=("mode","version"),dense=True))
    body=("事务隔离级别 锁等待 备份恢复 分区表 字符集 OBProxy 租户 会话 资源隔离 副本 合并 转储 " * 40)[:1800]
    rng=np.random.default_rng(3); V=rng.random((5000,DIM),dtype=np.float32); t0=time.time()
    for s in range(0,5000,1000):
        c.insert(n,[{"id":i,"text":f"文档{i} {body}","mode":"MySQL","version":"4.2.5","vector":V[i].tolist()} for i in range(s,s+1000)])
    ins=time.time()-t0
    ready(n,5000); ready(n,5000,"vector"); time.sleep(2)
    ts_s=[];ts_d=[]
    q=("锁等待的 SQL 怎么看",)
    for _ in range(10):
        t1=time.time(); c.search(n,data=["锁等待的 SQL 怎么看"],anns_field="sparse",limit=50); ts_s.append((time.time()-t1)*1000)
        t2=time.time(); c.search(n,data=[V[0].tolist()],anns_field="vector",limit=50); ts_d.append((time.time()-t2)*1000)
    return {"insert_5k_s":round(ins,1),"sparse_p50_ms":round(statistics.median(ts_s),1),
            "dense_p50_ms":round(statistics.median(ts_d),1),"data_dir_mb":dirsize(DIR),
            "bytes_per_chunk":round(dirsize(DIR)*1e6/5000,1),
            "est_25k_dense_sparse_mb":round(dirsize(DIR)/5000*25077,1)}

for i,(n,fn) in enumerate([("1_bm25_literal",f1),("2_bm25_scale_latency_size",f2),("3_idf_drift",f3),
                           ("4_bm25_filter",f4),("5_builtin_hybrid",f5),("6_upsert_delete",f6),
                           ("7_realistic_5k",f7)],1):
    rec(n,fn)
print(json.dumps(R,ensure_ascii=False,indent=1,default=str))
