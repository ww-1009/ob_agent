"""工具层共享数据模型、异常与协议接口。

所有 real/mock 实现都面向这里的数据模型编程，agent 不感知实现。
"""
from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

# ---------- 异常 ----------


class SqlExecutionError(ValueError):
    """SQL 执行层错误（连接失败、被拒绝、超时等）。"""


class OcpClientError(ValueError):
    """OCP 客户端错误（配置缺失 / 依赖缺失 / HTTP 失败统一入口）。"""


# ---------- SQL 结果模型 ----------

JsonValue = Any


class QueryResult(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[List[JsonValue]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False

    @model_validator(mode="after")
    def _sync_row_count(self) -> "QueryResult":
        # v1 语义：row_count 始终以实际 rows 长度为准，截断由 truncated 标志表达
        self.row_count = len(self.rows)
        return self


# ---------- 协议接口 ----------


@runtime_checkable
class OcpClient(Protocol):
    def get_slow_sql(self, cluster_id: int, tenant_id: int, start_time: str, end_time: str,
                     server_id: int = None, inner: bool = False, sql_text: str = None,
                     filter_expression: str = None, limit: int = None, sql_text_length: int = 100): ...

    def get_sql_text(self, cluster_id: int, tenant_id: int, sql_id: str, start_time: str, end_time: str): ...

    def get_sql_explain(self, cluster_id: int, tenant_id: int, uid: str, start_time: str, end_time: str): ...

    def get_sql_top_plan(self, cluster_id: int, tenant_id: int, sql_id: str, start_time: str, end_time: str): ...

    def get_tenants_list(self): ...

    def get_tenant_info(self, cluster_id, tenant_id): ...

    def get_clusters_list(self): ...

    def get_cluster_resource_stats(self, cluster_id): ...

    def get_server_resource_stats(self, cluster_id): ...


@runtime_checkable
class SqlExecutor(Protocol):
    def query(self, sql: str) -> QueryResult: ...

    def explain(self, sql: str) -> QueryResult: ...

    def table_ddl(self, table_name: str) -> QueryResult: ...
