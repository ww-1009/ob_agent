"""Mock OCP 客户端：读取 backend/data 下的 fixtures。"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

# OCP filter_expression 形如 "@avgElapsedTime > 300 and @executions > 100"；mock 仅实现首个条件
_FILTER_EXPR = re.compile(r"@([A-Za-z]+)\s*(>=|<=|>|<)\s*([0-9.]+)")


class MockOcpClient:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or _DEFAULT_DATA_DIR

    def _read(self, name: str) -> object:
        p = self._data_dir / name
        return json.loads(p.read_text(encoding="utf-8"))

    # ---- 新版工具面方法（固定夹具，离线 mock 用；字段与 OCP 真实报文同构）----

    def get_tenants_list(self):
        return self._read("ocp_tenants.json")

    def get_tenant_info(self, cluster_id, tenant_id):
        contents = self._read("ocp_tenants.json")
        for t in contents:
            if str(t.get("id")) == str(tenant_id):
                return t
        return {}

    def get_clusters_list(self):
        return self._read("ocp_clusters.json")

    def get_slow_sql(self, cluster_id, tenant_id, start_time, end_time,
                     server_id=None, inner=False, sql_text=None,
                     filter_expression=None, limit=None, sql_text_length=100):
        items = self._read("ocp_slow_sqls.json")
        if sql_text:
            kw = sql_text.lower()
            items = [it for it in items if kw in (it.get("sqlTextShort") or "").lower()]
        if inner:
            items = [it for it in items if it.get("inner")]
        else:
            items = [it for it in items if not it.get("inner")]
        if filter_expression:
            m = _FILTER_EXPR.search(filter_expression)
            if m:
                key, op, val = m.group(1), m.group(2), float(m.group(3))
                def _keep(it):
                    v = it.get(key)
                    if v is None:
                        return False
                    v = float(v)
                    return {">=": v >= val, ">": v > val, "<=": v <= val, "<": v < val}[op]
                items = [it for it in items if _keep(it)]
        if limit is not None:
            items = items[: int(limit)]
        out = []
        for it in items:
            it = dict(it)
            short = it.get("sqlTextShort") or ""
            if sql_text_length and len(short) > sql_text_length:
                it["sqlTextShort"] = short[:sql_text_length]
            out.append(it)
        return out

    def get_sql_text(self, cluster_id, tenant_id, sql_id, start_time, end_time):
        return self._read("ocp_sql_text.json")

    def get_sql_top_plan(self, cluster_id, tenant_id, sql_id, start_time, end_time):
        return self._read("ocp_top_plan.json")

    def get_sql_explain(self, cluster_id, tenant_id, uid, start_time, end_time):
        return self._read("ocp_sql_explain.json")

    def _cluster_ids(self) -> set[str]:
        return {str(c.get("id")) for c in self._read("ocp_clusters.json")}

    def get_cluster_resource_stats(self, cluster_id):
        """返回扁平的 ClusterResourceStats；集群不存在时返回 {}（工具层据此报 not_found）。"""
        if str(cluster_id) not in self._cluster_ids():
            return {}
        return self._read("ocp_cluster_stats.json")

    def get_server_resource_stats(self, cluster_id):
        """返回 ServerResourceStats 列表；集群不存在时返回 []。"""
        if str(cluster_id) not in self._cluster_ids():
            return []
        return self._read("ocp_server_stats.json")
