"""暴露给 langchain agent 的工具，薄封装。
- 工具内部异常一律转成 {"ok": false, "error": ..., "error_kind": ...} 的 JSON 观察结果，
  让 LLM 解释或换法重试（spec §8.1）；成功返回含 ok:true 的 JSON。
- 异常摘要会脱敏（主机/账号/连接串不下发），完整原文只进服务端日志，两者用 error_id 关联。
- execute_sql 先过 assert_read_only，禁止写操作。
"""
from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Sequence

from langchain_core.tools import BaseTool, tool
from langchain_community.agent_toolkits import FileManagementToolkit
from app.agent.tool_input import SlowSqlInput, FullSqlTextInput, SqlTopPlanInput, ExecuteSqlInput, SqlExplainInput, \
     TableDDLInput
from app.config import SqlConfig, load_settings
from app.tools.base import OcpClient, OcpClientError, SqlExecutionError, SqlExecutor
from app.tools.sql.guard import ReadOnlyViolation, assert_read_only
from app.tools.sql.mock import MockSqlExecutor
from app.tools.sql.oracle import OracleSqlExecutor, resolve_config as resolve_oracle_config
from app.tools.sql.mysql import MysqlSqlExecutor, resolve_config as resolve_mysql_config

logger = logging.getLogger(__name__)

# 下发给模型的错误摘要上限：驱动报错可能很长，超长文本对模型没有用
_MAX_ERROR_CHARS = 500
# 驱动/OCP 客户端的原始报错里可能夹带 host:port、IP、连接串账号。这些不该进入模型
# 上下文（agent 侧对自身异常同样只给通用文案，见 runner.py 的错误分支），只保留
# 「SQL 语义错误」这类能让模型自纠的信息。
_REDACTIONS = (
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b"), "<host>"),
    (re.compile(r"(?i)\b(password|passwd|pwd)\b\s*[=:]\s*\S+"), r"\1=<redacted>"),
    (re.compile(r"\b[A-Za-z0-9_.\-]+@[A-Za-z0-9_.\-]+(?:#[A-Za-z0-9_.\-]+)?"), "<account>"),
)
# 连接类失败只给分类：这类报错天然带环境信息，且模型重试也大概率无解
_CONNECTION_HINTS = (
    "connect", "connection", "timed out", "timeout", "refused", "unreachable",
    "broken pipe", "reset by peer", "ora-125", "ora-12170", "2003", "10060", "10061",
)


def _ok(**payload: object) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, default=str)


def _fail(error: str, *, kind: str = "error") -> str:
    return json.dumps({"ok": False, "error": error, "error_kind": kind}, ensure_ascii=False)


def _redact_error_text(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text[:_MAX_ERROR_CHARS]


def _classify_error(exc: BaseException) -> tuple[str, str]:
    """把异常映射成（error_kind, 可下发给模型的摘要）。"""
    if isinstance(exc, ReadOnlyViolation):
        # 只读违例是我们自己的中文说明，本身就是给模型的最优提示
        return "read_only", str(exc)
    if isinstance(exc, SqlExecutionError):
        text = str(exc) or type(exc).__name__
        if any(hint in text.lower() for hint in _CONNECTION_HINTS):
            return "connection", "数据库连接失败或查询超时（网络/鉴权/实例不可达），可稍后重试或换集群"
        # SQL 语义错误（未知列、语法错误等）必须保留：模型靠它换写法自纠；仅做脱敏
        return "sql", _redact_error_text(text)
    if isinstance(exc, OcpClientError):
        return "ocp", _redact_error_text(str(exc) or type(exc).__name__)
    if isinstance(exc, KeyError):
        return "payload", f"响应缺少字段: {exc}"
    return "internal", f"工具内部错误（{type(exc).__name__}）"


def _error(exc: BaseException) -> str:
    """工具内异常 → ok:false 观察结果：分类 + 脱敏 + 短 error_id。

    旧实现直接下发 ``str(e)``，会把驱动里的主机、账号、连接串片段一并交给模型。
    """
    error_id = uuid.uuid4().hex[:8]
    kind, message = _classify_error(exc)
    logger.warning("工具执行失败 [error_id=%s] %s: %s", error_id, type(exc).__name__, exc)
    return _fail(f"{message}（错误编号 {error_id}）", kind=kind)

def _create_db_connect(tenant_name: str, cluster_name: str, db_name: str, tenant_type: str, sql_config: SqlConfig):
    """按配置返回 SQL 执行器。

    - provider != "real"（即 mock）：直接用固定夹具的 MockSqlExecutor，使 mock 模式离线可用。
      mock 与方言无关，因此 Oracle 租户在 mock 下同样可用（便于演示与自测）。
    - provider == "real"：按 host_map 把「集群名 → host:port」解析成本租户的连接串，
      MySQL 模式走 PyMySQL（用户名 user@tenant#cluster），Oracle 模式走 OCI 驱动
      （租户信息走 DSN 的 service_name，用户名 user@tenant）。
    解析失败抛 SqlExecutionError，由调用方的 try 转成 {"ok": false} 观察结果。
    """
    if tenant_type not in {"MYSQL", "ORACLE"}:
        raise SqlExecutionError(f"未知租户类型: {tenant_type}（可选 MYSQL | ORACLE）")

    if sql_config.provider != "real":
        return MockSqlExecutor()

    ob_host_str = sql_config.host
    if not ob_host_str:
        raise SqlExecutionError("sql_ro.host_map 未配置：real 模式需要「集群名 → host:port」映射")
    host_map = ast.literal_eval(ob_host_str)
    entry = host_map.get(cluster_name)
    if not entry:
        raise SqlExecutionError(
            f"sql_ro.host_map 中没有集群 {cluster_name}（已有: {', '.join(sorted(host_map)) or '空'}）"
        )
    host, _, port = str(entry).partition(':')
    common = dict(
        provider="real",
        host=host,
        port=int(port) if port else sql_config.port,
        db_name=db_name,
        password=sql_config.password,
        connect_timeout=sql_config.connect_timeout,
        query_timeout_seconds=sql_config.query_timeout_seconds,
        max_rows=sql_config.max_rows,
    )

    if tenant_type == "ORACLE":
        oracle_cfg = SqlConfig(
            **common,
            # 基础账号：Oracle 由 resolve_config 补成 user@tenant；漏掉会让用户名变成 "@租户"
            username=sql_config.username,
            driver=sql_config.driver,
            service_name=sql_config.service_name,
        )
        # Oracle 模式：租户走 service_name，用户名补成 user@tenant
        return OracleSqlExecutor(resolve_oracle_config(oracle_cfg, tenant_name))

    # MySQL 模式：租户与集群写在用户名里 user@tenant#cluster
    return MysqlSqlExecutor(resolve_mysql_config(
        SqlConfig(**common, username=sql_config.username),
        tenant_name,
        cluster_name,
    ))

def build_tools(
    ocp: OcpClient,
    sql_ro_config: SqlConfig,
    *,
    send_row_data: bool = True,
) -> Sequence[BaseTool]:

    # 初始化文件管理 Toolkit，指定根目录和所需工具
    file_tools = FileManagementToolkit(
        root_dir="./ob_wiki",  # 限制 Agent 只能在该目录下读写文件，保障安全性
        selected_tools=[
            "read_file",
            "list_directory",
        ]
    ).get_tools()

    # 真实执行器按 (租户, 集群, 库, 租户类型) 缓存：复用底层长连接，
    # 避免每次工具调用都重新建连与鉴权（见 PooledSqlExecutor 的连接复用说明）。
    executor_cache: dict[tuple[str, str, str, str], SqlExecutor] = {}

    def _db_connect(tenant_name: str, cluster_name: str, db_name: str, tenant_type: str):
        key = (tenant_name, cluster_name, db_name, tenant_type)
        executor = executor_cache.get(key)
        if executor is None:
            executor = _create_db_connect(tenant_name, cluster_name, db_name, tenant_type, sql_ro_config)
            executor_cache[key] = executor
        return executor

    @tool
    def get_tenant_info() -> str:
        """
        获取所有租户信息
        """
        target_keys = {"id", "clusterId", "clusterName", "obTenantId", "description", "mode", "name"}
        try:
            tenant_list = ocp.get_tenants_list()
            filtered_items: list[dict] = []
            for item in tenant_list:
                if not isinstance(item, dict):
                    continue
                # 单次遍历：过滤的同时把原始字段 id 重命名为 tenantId，避免给 LLM 的「id」含义含糊
                out = {k: v for k, v in item.items() if k in target_keys}
                if "id" in out:
                    out["tenantId"] = out.pop("id")
                filtered_items.append(out)
            return _ok(items=filtered_items)
        except Exception as e:
            return _error(e)


    @tool(args_schema=SlowSqlInput)
    def get_slow_sql(cluster_id: int, tenant_id: int,start_time: str,end_time: str,sql_text_length: int,server_id: int, inner: bool,
                       sql_text: str, filter_expression: str, limit: int ) -> str:
        """
        获取一段时间内的慢sql列表和慢sql运行情况
        """
        target_keys = [
            "sqlId","avgAffectedRows", "avgCpuTime", "avgDiskReads", "avgElapsedTime",
            "avgExecuteTime", "avgExecutorRpcCount", "avgLogicalReads", "avgMemstoreReadRows", "avgMemstoreReadRows",
            "avgNetWaitTime", "avgPartitionCount", "avgQueueTime", "avgReturnRows",
            "avgSsstoreReadRows", "avgWaitTime", "dbName", "executions",
            "failCount", "sumElapsedTime", "tableScanPercentage", "tenantName", "userName", "sqlTextShort"
        ]
        try:
            items = ocp.get_slow_sql(cluster_id, tenant_id, start_time, end_time, server_id, inner, sql_text, filter_expression,limit, sql_text_length)
            filtered_items = [
                {k: v for k, v in item.items() if k in target_keys}
                for item in items
                if isinstance(item, dict)
            ]
            return _ok(items=filtered_items)
        except Exception as e:
            return _error(e)

    @tool(args_schema=FullSqlTextInput)
    def get_full_sql_text(cluster_id: int, tenant_id: int, sql_id: str,start_time: str,end_time: str) -> str:
        """
        获取完整sql文本,查询的库和表
        """
        try:
            data = ocp.get_sql_text(cluster_id, tenant_id, sql_id, start_time, end_time)
            sql_text = data["fulltext"]
            db_name = data["dbName"]
            tables = data["tables"]
            return _ok(items=[{"sqlText":sql_text,"dbName":db_name,"tables":tables}])
        except Exception as e:
            return _error(e)

    @tool(args_schema=SqlTopPlanInput)
    def get_sql_top_plan(cluster_id: int, tenant_id: int, sql_id: str,start_time: str, end_time: str) -> str:
        """
        获取指定 SQL ID 的计划uid和该执行计划的执行情况
        """
        try:
            sql_top_plan_list = ocp.get_sql_top_plan(cluster_id, tenant_id, sql_id, start_time, end_time)
            if not sql_top_plan_list:
                # 「查不到」不是成功：旧实现回 {"ok": true, "message": "no data"}，
                # 审计里会被记成一次成功调用，模型也容易当成「已拿到计划」继续往下推。
                return _fail("未获取到该 SQL 的计划 uid（可能未被采样到）", kind="not_found")
            else:
                summary_lines = []
                for idx, data in enumerate(sql_top_plan_list, start=1):
                    uid = data.get("uid")
                    avg_elapsed_time = data["avgElapsedTime"]
                    first_load_time = data["firstLoadTime"]
                    summary_lines.append(
                        {"sql_id": sql_id, "uid": uid, "avg_elapsed_time": avg_elapsed_time,"first_load_time":first_load_time}
                    )
                return _ok(items=summary_lines)
        except Exception as e:
            return _error(e)

    @tool(args_schema=SqlExplainInput)
    def get_sql_explain(cluster_id: int, tenant_id: int, uid: str,start_time: str,end_time: str) -> str:
        """
        根据 SQL ID 和 uid 获取执行计划结构数据，数据包括：执行计划算子的数组、操作列表的数组和操作算子的数组
        """
        try:
            data_list = ocp.get_sql_explain(cluster_id, tenant_id, uid, start_time, end_time)
            if not data_list:
                # 同上：空计划必须是 ok:false。旧实现返回裸文本
                # "未获取到执行计划结构数据\n"，tool_trace 对非 JSON 且不以
                # Error: 开头的文本一律判成功（tool_trace.py:62-69）。
                return _fail("未获取到执行计划结构数据（该 uid 可能已过期）", kind="not_found")
            else:
                plan_data = data_list["data"]
                plan_operation_summery = data_list["planOperationSummery"]
                root_operations = data_list["rootOperations"]
                return _ok(items=[{"plan_data": plan_data, "plan_operation_summery": plan_operation_summery, "root_operations": root_operations}])
        except Exception as e:
            return _error(e)

    @tool(args_schema=ExecuteSqlInput)
    def execute_sql(tenant_name: str, cluster_name: str, db_name: str, sql: str, tenant_type: str) -> str:
        """连接数据库执行查询操作"""
        # assert_read_only 与连接解析都必须在 try 内：否则只读违例（写 SQL）、host_map
        # 缺键等异常会抛出工具之外，违反「工具异常一律转 ok:false」的约定并中断 agent 轮次。
        try:
            assert_read_only(sql)
            sql_executor = _db_connect(tenant_name, cluster_name, db_name, tenant_type)
            r = sql_executor.query(sql)
            rows = [] if not send_row_data else r.rows
            return _ok(columns=r.columns, rows=rows, row_count=r.row_count, truncated=r.truncated)
        except Exception as e:
            return _error(e)

    @tool(args_schema=TableDDLInput)
    def get_table_ddl(tenant_name: str, cluster_name: str, db_name: str, tenant_type: str,table_name: str) -> str:
        """查询表结构与索引"""
        # 方言差异（MySQL 的 SHOW CREATE TABLE / Oracle 的 DBMS_METADATA）由执行器
        # 自己决定，这里不再按 tenant_type 分支。
        try:
            sql_executor = _db_connect(tenant_name, cluster_name, db_name, tenant_type)
            r = sql_executor.table_ddl(table_name)
            return _ok(columns=r.columns, rows=r.rows, row_count=r.row_count, truncated=r.truncated)
        except Exception as e:
            return _error(e)

    return [get_tenant_info, get_slow_sql,get_full_sql_text, get_sql_top_plan,get_sql_explain, execute_sql,get_table_ddl]+file_tools

if __name__ == '__main__':
    settings = load_settings()
    sql_config = settings.sql_ro
    db = _create_db_connect('uat1006', 'test_shrbank_ob002', 'dckkdb', 'MYSQL', sql_config)
    data = db.query("show create table user_sign_info;")
    print(data)