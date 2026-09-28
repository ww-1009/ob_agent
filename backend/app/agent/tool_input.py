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
    db_name: str = Field(description="数据库名（Oracle 模式租户：该值同时作为连接的 service name，须与该租户的 SERVICE_NAME 一致）")
    sql: str = Field(description="需要执行的查询SQL")
    tenant_type: Literal["MYSQL", "ORACLE"] = Field(description="租户类型。MYSQL 用 LIMIT/反引号/SHOW CREATE TABLE；ORACLE 用 FETCH FIRST/无引号/不加 LIMIT")


class SqlExplainInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    uid: str = Field(description="计划uid")
    start_time: str = Field(description=_START_DESC, default_factory=_window_start_iso)
    end_time: str = Field(description=_END_DESC, default_factory=_now_iso)


class PlanCompareInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    uid_before: str = Field(
        description="基准计划的 uid（改动前/上一次的那份，由 get_sql_top_plan 的 items 里取）"
    )
    uid_after: str = Field(
        description="目标计划的 uid（改动后/本次的那份，由 get_sql_top_plan 的 items 里取）"
    )
    start_time: str = Field(description=_START_DESC, default_factory=_window_start_iso)
    end_time: str = Field(description=_END_DESC, default_factory=_now_iso)
    start_time_before: Optional[str] = Field(
        description="基准计划的起始时间；两份计划采集时间不同时才需要单独指定，不传则用 start_time",
        default=None,
    )
    end_time_before: Optional[str] = Field(
        description="基准计划的结束时间；不传则用 end_time", default=None
    )
    start_time_after: Optional[str] = Field(
        description="目标计划的起始时间；不传则用 start_time", default=None
    )
    end_time_after: Optional[str] = Field(
        description="目标计划的结束时间；不传则用 end_time", default=None
    )


class TableDDLInput(BaseModel):
    cluster_name: str = Field(description="集群名")
    tenant_name: str = Field(description="租户名")
    db_name: str = Field(description="数据库名（Oracle 模式租户：该值同时作为连接的 service name，须与该租户的 SERVICE_NAME 一致）")
    tenant_type: Literal["MYSQL", "ORACLE"] = Field(description="租户类型。MYSQL 走 SHOW CREATE TABLE；ORACLE 走 DBMS_METADATA.GET_DDL")
    table_name: str = Field(description="表名")


class ClusterIdInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID（由 get_cluster_list 获取）")


class DocSearchInput(BaseModel):
    query: str = Field(
        description="检索词或问题。中文优先用 2-4 字的核心词（如「锁等待」「资源水位」「事务隔离级别」），"
                    "多个词用空格分隔；不要带「怎么/如何/什么」这类疑问词"
    )
    limit: Optional[int] = Field(description="返回的相关小节数量，默认 5，最大 20", default=5)
    mode: Optional[str] = Field(
        description="按租户模式过滤：MYSQL 或 ORACLE。不填时若提问里明确写了模式会自动识别；"
                    "文档库中 MySQL/Oracle 模式有同名文档，提问涉及模式时务必过滤",
        default=None
    )
    version: Optional[str] = Field(
        description="按 OceanBase 版本过滤，如 4.2.5。不填时若提问里写了版本号会自动识别",
        default=None
    )
    include_index: Optional[bool] = Field(
        description="问「有哪些 / 包含哪些 / 怎么分类」这类要清单的问题时置 true，让分类索引页与"
                    "知识库检索指南（README）正常参与检索；回答仍以正文为准",
        default=False
    )


class DocReadInput(BaseModel):
    path: str = Field(
        description="文档路径，取自 search_docs 返回的 path（相对 ./doc，形如 "
                    "ob_wiki/OceanBase 数据库/参考指南/…/合并异常问题排查.md）"
    )
    section: Optional[str] = Field(
        description="小节名或关键词（如「典型案例」「设置方法」）；不填则返回文档开头，"
                    "响应里的 sections 字段可用于挑选小节",
        default=None
    )
    max_chars: Optional[int] = Field(
        description="最多返回的字符数，默认 6000，最大 20000", default=None
    )
