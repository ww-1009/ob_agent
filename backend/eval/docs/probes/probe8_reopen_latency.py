"""probe8：milvus-lite v3 的检索延迟必须在**新进程 reopen** 之后测。

为什么单列一个探针：probe7 的第 9 段是在**建索引的那个进程里**量的，2 vCPU 上得到
sparse P50 779 ms / dense 599 ms；close 之后在另一个进程里 reopen 同一个 data_dir，
同一批数据只有 sparse 109 ms / dense 27 ms——约 10x 的差距是构建期伪影（segment 还在
growing/未压实）。本探针把正确口径固化成可重跑脚本，并在同一份产物里同时记录伪影读数，
方便以后有人再看到那些大数字时立刻知道原因。

用法：
    ./backend/.venv/bin/python backend/eval/docs/probes/probe8_reopen_latency.py
产物：同目录 probe8.out.json（可用 PROBE8_DIR 换 data_dir，默认 /tmp/milvus_spike/data8.db）

注意：测量期间机器上不要有别的重活。2 vCPU 上并发两个 Milvus 进程也会把 P50 从 ~80 ms 推到 ~800 ms。
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

sys.path.insert(0, str(Path(__file__).parent))

import probe7_lite_nullable_linux as p7  # noqa: E402

ROWS = 25000
BATCH = 2000
DIR = os.environ.get("PROBE8_DIR", "/tmp/milvus_spike/data8.db")
OUT = Path(__file__).parent / "probe8.out.json"
QUERY = "锁等待的 SQL 怎么看"
VOCAB = [
    "锁等待", "事务隔离", "备份恢复", "分区表", "字符集", "OBProxy", "租户", "会话",
    "SQL 审计", "资源隔离", "副本", "合并", "系统变量", "执行计划", "慢查询", "连接池",
]

# ---------------------------------------------------------------------------
# 子进程部分：只负责 reopen 已有 data_dir 并计时。
REOPEN_SRC = r'''
import json, os, random, statistics, sys, time

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

sys.path.insert(0, os.environ["PROBE8_PROBES"])
from probe7_lite_nullable_linux import dirsize  # noqa: E402

DIR = sys.argv[1]
QUERY = "锁等待的 SQL 怎么看"

t0 = time.time()
client = MilvusClient(DIR)
name = [n for n in client.list_collections() if n != "ob_meta"][0]
dense_field = sparse_field = dim = None
for f in client.describe_collection(name)["fields"]:
    if f["type"] == 101:  # FLOAT_VECTOR
        dense_field, dim = f["name"], int(f["params"]["dim"])
    elif f["type"] == 104:  # SPARSE_FLOAT_VECTOR
        sparse_field = f["name"]
client.load_collection(name)
open_load_s = time.time() - t0

rnd = random.Random(99999)
qv = [rnd.random() for _ in range(dim)]


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
    AnnSearchRequest(data=[QUERY], anns_field=sparse_field, param={}, limit=50),
    AnnSearchRequest(data=[qv], anns_field=dense_field, param={}, limit=50),
]
out = {
    "pid": os.getpid(),
    "collection": name,
    "rows": client.get_collection_stats(name).get("row_count"),
    "fields": {"sparse": sparse_field, "dense": dense_field},
    "open_and_load_s": round(open_load_s, 2),
    "sparse_l50": bench(lambda: client.search(name, data=[QUERY], anns_field=sparse_field, limit=50)),
    "dense_l50": bench(lambda: client.search(name, data=[qv], anns_field=dense_field, limit=50)),
    "hybrid_builtin_l10": bench(
        lambda: client.hybrid_search(name, reqs=reqs, ranker=RRFRanker(60), limit=10)
    ),
    "desc_dense": {
        k: v for k, v in client.describe_index(name, dense_field).items()
        if k in ("index_type", "metric_type", "state", "total_rows")
    },
    "data_dir_mb": dirsize(DIR),
}
client.close()
print(json.dumps(out, ensure_ascii=False, default=str))
'''


def _release(path: str) -> None:
    """显式释放本进程对 data_dir 的持有（``MilvusClient.close()`` 不释放 flock）。"""
    from milvus_lite.server_manager import server_manager_instance

    server_manager_instance.release_server(str(Path(path).absolute()))


def build() -> dict:
    """新建 data_dir 并插入 ROWS 行（精确 25000，不重犯 probe7 f9 的 26000 行失误）。"""
    shutil.rmtree(DIR, ignore_errors=True)
    client = p7.MilvusClient(DIR)
    name = "big25k"
    client.create_collection(name, schema=p7.schema(extra=("kind",)), index_params=p7.ip_for(client, extra=("kind",)))
    body = "。".join(VOCAB * 12)[:1750]
    t0 = time.time()
    for start in range(0, ROWS, BATCH):
        client.insert(
            name,
            [
                {"id": i, "text": f"文档{i} {body}", "kind": "doc", "vector": p7.rng_vec(i)}
                for i in range(start, min(start + BATCH, ROWS))
            ],
        )
    insert_s = time.time() - t0
    qv = p7.rng_vec(99999)
    p7.settle(client, name, 10.0, ("sparse", "vector"), warm_data=qv)

    import resource  # noqa: PLC0415

    def bench(fn, reps=6, warm=2):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            t = time.time()
            fn()
            ts.append((time.time() - t) * 1000)
        return {"p50": round(statistics.median(ts), 1), "min": round(min(ts), 1), "max": round(max(ts), 1)}

    artifact = {
        "rows": client.get_collection_stats(name).get("row_count"),
        "insert_s": round(insert_s, 1),
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "sparse_l50": bench(lambda: client.search(name, data=[QUERY], anns_field="sparse", limit=50)),
        "dense_l50": bench(lambda: client.search(name, data=[qv], anns_field="vector", limit=50)),
        "data_dir_mb": p7.dirsize(DIR),
    }
    client.close()
    _release(DIR)  # 不释放的话子进程 reopen 会撞 DataDirLockedError（本探针第一版就踩了）
    return artifact


def reopen() -> dict:
    env = {**os.environ, "PROBE8_PROBES": str(Path(__file__).parent)}
    proc = subprocess.run(
        [sys.executable, "-c", REOPEN_SRC, DIR], capture_output=True, text=True, env=env, check=False
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()[-2000:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    result: dict = {"dir": DIR, "rows_target": ROWS}
    result["build_and_in_process"] = build()
    result["reopened"] = reopen()
    result["verdict"] = (
        "构建进程内的读数（build_and_in_process）是伪影，不要当基线；"
        "权威基线取 reopened（close 后新进程 reopen 同一 data_dir）。"
    )
    payload = json.dumps(result, ensure_ascii=False, indent=1)
    OUT.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()