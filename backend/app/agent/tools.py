"""暴露给 langchain agent 的工具，薄封装。
- 工具内部异常一律转成 {"ok": false, "error": ...} 的 JSON 观察结果，
  让 LLM 解释或换法重试（spec §8.1）；成功返回含 ok:true 的 JSON。
- execute_sql 先过 assert_read_only，禁止写操作。
"""
from __future__ import annotations

import json
import ast
from typing import Sequence

from langchain_core.tools import BaseTool, tool
from langchain_community.agent_toolkits import FileManagementToolkit
from app.agent.tool_input import SlowSqlInput, FullSqlTextInput, SqlTopPlanInput, ExecuteSqlInput, SqlExplainInput, \
     TableDDLInput
from app.config import SqlConfig, load_settings
from app.tools.base import OcpClient, SqlExecutor
from app.tools.sql.guard import assert_read_only
from app.tools.sql.real import RealSqlExecutor


def _ok(**payload: object) -> str:
    return json.dumps({"ok": True, **payload}, ensure_ascii=False, default=str)


def _fail(error: str) -> str:
    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)

def _create_db_connect(tenant_name: str, cluster_name: str, db_name: str, tenant_type: str, sql_config: SqlConfig):
    user=sql_config.username

    password=sql_config.password
    ob_host_str=sql_config.host
    if ob_host_str:
        config = ast.literal_eval(ob_host_str)
        host,port = config[cluster_name].split(':')
    else:
        return None
    user_str = f'{user}@{tenant_name}#{cluster_name}'
    config = SqlConfig(provider="real",host=host,db_name=db_name,username=user_str,password=password,
                       connect_timeout=sql_config.connect_timeout,query_timeout_seconds=sql_config.query_timeout_seconds)
    if tenant_type == "MYSQL":
        return RealSqlExecutor(config)
    else:
        # todo:待完善oracle租户数据库连接
        return None

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
            return _fail(str(e) or type(e).__name__)


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
            return _fail(str(e) or type(e).__name__)

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
            return _fail(str(e) or type(e).__name__)

    @tool(args_schema=SqlTopPlanInput)
    def get_sql_top_plan(cluster_id: int, tenant_id: int, sql_id: str,start_time: str, end_time: str) -> str:
        """
        获取指定 SQL ID 的计划uid和该执行计划的执行情况
        """
        try:
            sql_top_plan_list = ocp.get_sql_top_plan(cluster_id, tenant_id, sql_id, start_time, end_time)
            if not sql_top_plan_list:
                return _ok(items=[{"message":"no data"}])
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
            return _fail(str(e) or type(e).__name__)

    @tool(args_schema=SqlExplainInput)
    def get_sql_explain(cluster_id: int, tenant_id: int, uid: str,start_time: str,end_time: str) -> str:
        """
        根据 SQL ID 和 uid 获取执行计划结构数据，数据包括：执行计划算子的数组、操作列表的数组和操作算子的数组
        """
        try:
            data_list = ocp.get_sql_explain(cluster_id, tenant_id, uid, start_time, end_time)
            if not data_list:
                return "未获取到执行计划结构数据\n"
            else:
                plan_data = data_list["data"]
                plan_operation_summery = data_list["planOperationSummery"]
                root_operations = data_list["rootOperations"]
                return _ok(items=[{"plan_data": plan_data, "plan_operation_summery": plan_operation_summery, "root_operations": root_operations}])
        except Exception as e:
            return _fail(str(e) or type(e).__name__)

    @tool(args_schema=ExecuteSqlInput)
    def execute_sql(tenant_name: str, cluster_name: str, db_name: str, sql: str, tenant_type: str) -> str:
        """连接数据库执行查询操作"""
        assert_read_only(sql)
        sql_executor = _create_db_connect(tenant_name, cluster_name, db_name, tenant_type,sql_ro_config)
        # todo:待完善oracle租户数据库连接
        if sql_executor is None:
            # logger.warning(f"oracle租户连接功能待开发")
            return "暂不支持查询oracle租户"
        # logger.info(f"Calling tool: execute_sql  with arguments: {sql}")
        try:
            r = sql_executor.query(sql)
            rows = [] if not send_row_data else r.rows
            return _ok(columns=r.columns, rows=rows, row_count=r.row_count, truncated=r.truncated)
        except Exception as e:
            return _fail(str(e) or type(e).__name__)

    @tool(args_schema=TableDDLInput)
    def get_table_ddl(tenant_name: str, cluster_name: str, db_name: str, tenant_type: str,table_name: str) -> str:
        """查询表结构与索引"""
        try:
            # logger.info(f"Calling tool: execute_sql  with arguments: {sql}")
            if tenant_type == "MYSQL":
                sql_executor = _create_db_connect(tenant_name, cluster_name, db_name, tenant_type, sql_ro_config)
                r = sql_executor.query(f"show create table {table_name};")
                return _ok(columns=r.columns, rows=r.rows, row_count=r.row_count, truncated=r.truncated)
            else:
                # todo:待完善oracle租户数据库连接
                # logger.warning(f"oracle租户连接功能待开发")
                return "暂不支持查询oracle租户"
        except Exception as e:
            return _fail(str(e) or type(e).__name__)

    return [get_tenant_info, get_slow_sql,get_full_sql_text, get_sql_top_plan,get_sql_explain, execute_sql,get_table_ddl]+file_tools

if __name__ == '__main__':
    settings = load_settings()
    sql_config = settings.sql_ro
    db = _create_db_connect('uat1006', 'test_shrbank_ob002', 'dckkdb', 'MYSQL', sql_config)
    data = db.query("show create table user_sign_info;")
    print(data)