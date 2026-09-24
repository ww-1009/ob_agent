"""工具的名词标签：供工具轨迹与审计展示（进度语仍在 runner.TOOL_STATUS）。

单独成模块是为了让 runner 与 confirm 都能引用而不产生循环依赖。
"""
from __future__ import annotations

TOOL_LABEL: dict[str, str] = {
    "get_tenant_info": "获取租户信息",
    "get_cluster_list": "获取集群列表",
    "get_cluster_resource_stats": "获取集群资源水位",
    "get_server_resource_stats": "获取OBServer资源水位",
    "get_slow_sql": "拉取慢SQL列表",
    "get_full_sql_text": "拉取完整SQL文本",
    "get_sql_top_plan": "拉取SQL计划uid",
    "get_sql_explain": "拉取执行计划结构",
    "execute_sql": "执行只读 SQL",
    "get_table_ddl": "获取表结构",
    "read_file": "读取官方文档",
    "list_directory": "列出文档目录",
}


def tool_label(name: str) -> str:
    return TOOL_LABEL.get(name, name or "未知工具")
