from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class TenantInfoInput(BaseModel):
    tenant_name: str = Field(description="租户名")

class SlowSqlInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    start_time: str = Field(
        description="查看慢 SQL 历史参数的起始时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=(datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=30)).isoformat(timespec='seconds')
    )
    end_time: str = Field(
        description="查看慢 SQL 历史参数的结束时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')
    )
    server_id : Optional[int] = Field(
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
    start_time: str = Field(
        description="查看SQL的起始时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=(datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=30)).isoformat(timespec='seconds')
    )
    end_time: str = Field(
        description="查看慢 SQL 历史参数的结束时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')
    )


class SqlTopPlanInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    sql_id: str = Field(description="sql ID")
    start_time: str = Field(
        description="查看SQL的起始时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=(datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=30)).isoformat(timespec='seconds')
    )
    end_time: str = Field(
        description="查看慢 SQL 历史参数的结束时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')
    )


class SqlExplainToolInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    uid: str = Field(description="计划uid")
    start_time: str = Field(
        description="查看SQL的起始时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=(datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=30)).isoformat(timespec='seconds')
    )
    end_time: str = Field(
        description="查看慢 SQL 历史参数的结束时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')
    )

class ExecuteSqlInput(BaseModel):
    cluster_name: str = Field(description="集群名")
    tenant_name: str = Field(description="租户名")
    db_name: str = Field(description="数据库名")
    sql: str = Field(description="需要执行的查询SQL")
    tenant_type: Literal["MYSQL","ORACLE"] = Field(description="租户类型")


class SqlExplainInput(BaseModel):
    cluster_id: int = Field(description="集群的 ID")
    tenant_id: int = Field(description="租户的 ID")
    uid: str = Field(description="计划uid")
    start_time: str = Field(
        description="查看SQL的起始时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=(datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=30)).isoformat(timespec='seconds')
    )
    end_time: str = Field(
        description="查看慢 SQL 历史参数的结束时间。该时间格式为：2020-02-16T05:32:16+08:00",
        default=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')
    )

class TableDDLInput(BaseModel):
    cluster_name: str = Field(description="集群名")
    tenant_name: str = Field(description="租户名")
    db_name: str = Field(description="数据库名")
    tenant_type: Literal["MYSQL","ORACLE"] = Field(description="租户类型")
    table_name: str = Field(description="表名")