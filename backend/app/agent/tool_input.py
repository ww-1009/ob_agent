"""工具的入参模型。

时间窗默认值必须用 default_factory：写成 `default=datetime.now(...)` 会在**模块导入时**
求值一次并被固化为常量，进程存活越久，默认窗口偏离「现在」越远（曾导致长时间运行后
不带时间地问「有哪些慢SQL？」查到的是启动时刻那 30 分钟，静默返回空结果）。
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("Asia/Shanghai")
_DEFAULT_WINDOW_MINUTES = 30

_TIME_FORMAT_HINT = "该时间格式为：2020-02-16T05:32:16+08:00"


def _now_iso() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


def _window_start_iso() -> str:
    return (datetime.now(_TZ) - timedelta(minutes=_DEFAULT_WINDOW_MINUTES)).isoformat(timespec="seconds")


# 默认值不会出现在 JSON Schema 里，因此把默认窗口写进 description 让模型知道可以省略
_START_DESC = f"查看SQL的起始时间。{_TIME_FORMAT_HINT}；不传时默认取当前时间前 {_DEFAULT_WINDOW_MINUTES} 分钟"
_END_DESC = f"查看SQL的结束时间。{_TIME_FORMAT_HINT}；不传时默认取当前时间"


class TenantInfoInput(BaseModel):
    tenant_name: str = Field(description="租户名")


class SlowSqlInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    start_time: str = Field(description=_START_DESC, default_factory=_window_start_iso)
    end_time: str = Field(description=_END_DESC, default_factory=_now_iso)
    server_id: Optional[int] = Field(
        description="查询在指定 OceanBase 服务器上的计划的性能。不指定时，查询 SQL 在所有服务器上的计划的性能",
        default=None
    )
    inner: Optional[bool] = Field(
        description="是否为内部 SQL",
        default=False
    )
    sql_text: Optional[str] = Field(
        description="SQL 包含的关键词，关键词不区分大小写",
        default=None
    )
    filter_expression: Optional[str] = Field(
        description="filter_expression:通过 @ 来引用过滤返回数据，可选字段[cpuPercentage,executions,avgElapsedTime,failCount]，例如'@avgElapsedTime > 300 and @executions > 100'表示平均响应时间大于300毫秒并且执行次数大于100",
        default=None
    )
    limit: Optional[int] = Field(
        description="返回的 TOP 数目",
        default=5
    )
    sql_text_length: Optional[int] = Field(
        description="返回 SQL 文本的最大长度",
        default=100
    )


class FullSqlTextInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    sql_id: str = Field(description="sql ID")
    start_time: str = Field(description=_START_DESC, default_factory=_window_start_iso)
    end_time: str = Field(description=_END_DESC, default_factory=_now_iso)


class SqlTopPlanInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    sql_id: str = Field(description="sql ID")
    start_time: str = Field(description=_START_DESC, default_factory=_window_start_iso)
    end_time: str = Field(description=_END_DESC, default_factory=_now_iso)


class ExecuteSqlInput(BaseModel):
    cluster_name: str = Field(description="集群名")
    tenant_name: str = Field(description="租户名")
    db_name: str = Field(description="数据库名")
    sql: str = Field(description="需要执行的查询SQL")
    tenant_type: Literal["MYSQL", "ORACLE"] = Field(description="租户类型。MYSQL 用 LIMIT/反引号/SHOW CREATE TABLE；ORACLE 用 FETCH FIRST/无引号/不加 LIMIT")


class SqlExplainInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    uid: str = Field(description="计划uid")
    start_time: str = Field(description=_START_DESC, default_factory=_window_start_iso)
    end_time: str = Field(description=_END_DESC, default_factory=_now_iso)


class TableDDLInput(BaseModel):
    cluster_name: str = Field(description="集群名")
    tenant_name: str = Field(description="租户名")
    db_name: str = Field(description="数据库名")
    tenant_type: Literal["MYSQL", "ORACLE"] = Field(description="租户类型。MYSQL 走 SHOW CREATE TABLE；ORACLE 走 DBMS_METADATA.GET_DDL")
    table_name: str = Field(description="表名")
