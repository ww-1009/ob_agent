"""精简后的核心通路测试：只覆盖当前 API 的关键行为。

替代已删除的 test_agent_tools.py / test_real_ocp.py / test_real_sql.py 中仍有效的那部分
契约：工具注册名、工具层只读守卫、real 层纯函数、MysqlSqlExecutor 连接前拒绝写 SQL，
以及 MySQL / Oracle 两个真实执行器的命名与结构对称性。
"""
import json
from abc import ABC

import pytest

from app.agent.tools import _classify_error, _create_db_connect, _error, build_tools
from app.config import SqlConfig
from app.tools.base import OcpClientError, SqlExecutionError
from app.tools.ocp.mock import MockOcpClient
from app.tools.ocp.real import _elapsed_us, _first_present, _ms_to_us, _normalize_mode
from app.tools.sql.base import PooledSqlExecutor
from app.tools.sql.guard import ReadOnlyViolation
from app.tools.sql.oracle import (
    OracleSqlExecutor,
    build_dsn as build_oracle_dsn,
    resolve_config as resolve_oracle_config,
)
from app.tools.sql.mysql import MysqlSqlExecutor, resolve_config as resolve_mysql_config

_EXPECTED_TOOLS = {
    "get_tenant_info",
    "get_slow_sql",
    "get_full_sql_text",
    "get_sql_top_plan",
    "get_sql_explain",
    "execute_sql",
    "get_table_ddl",
    "read_file",
    "list_directory",
}

_EXEC_ARGS = {
    "cluster_name": "c1",
    "tenant_name": "t1",
    "db_name": "shop",
    "tenant_type": "MYSQL",
}


@pytest.fixture
def tools():
    # provider=mock 时 execute_sql/get_table_ddl 内部自动使用 MockSqlExecutor
    built = build_tools(MockOcpClient(), SqlConfig(provider="mock"), send_row_data=True)
    return {t.name: t for t in built}


def test_build_tools_registers_expected_names(tools):
    assert set(tools) == _EXPECTED_TOOLS


def test_tool_metadata_is_in_sync_with_registered_tools():
    """TOOL_STATUS / TOOL_LABEL 必须跟着工具注册走：漏一处就少进度语或标签。

    这几份清单原先靠人工同步（prompt 里的工具说明、runner.TOOL_STATUS、
    tool_labels.TOOL_LABEL、confirm.CONFIRM_TOOL_LABELS），容易漂移；
    这里把「等于注册集合」和「审批只覆盖执行 SQL」固化下来。
    """
    from app.agent.confirm import CONFIRM_TOOL_LABELS
    from app.agent.runner import TOOL_STATUS
    from app.agent.tool_labels import TOOL_LABEL

    assert set(TOOL_STATUS) == _EXPECTED_TOOLS
    assert set(TOOL_LABEL) == _EXPECTED_TOOLS
    # 审批只覆盖执行 SQL：OCP 元数据与文档读取不需要人工确认
    assert set(CONFIRM_TOOL_LABELS) == {"execute_sql"}


def test_get_tenant_info_ok(tools):
    data = json.loads(tools["get_tenant_info"].invoke({}))
    assert data["ok"] is True
    assert isinstance(data["items"], list) and data["items"]


def test_get_slow_sql_respects_limit(tools):
    data = json.loads(
        tools["get_slow_sql"].invoke({"cluster_id": 1, "tenant_id": 1001, "limit": 2})
    )
    assert data["ok"] is True
    assert len(data["items"]) == 2


def test_execute_sql_read_query_ok(tools):
    data = json.loads(
        tools["execute_sql"].invoke({**_EXEC_ARGS, "sql": "select * from orders"})
    )
    assert data["ok"] is True
    assert data["columns"]


def test_execute_sql_write_is_rejected(tools):
    data = json.loads(
        tools["execute_sql"].invoke({**_EXEC_ARGS, "sql": "delete from orders"})
    )
    assert data["ok"] is False and data["error"]


def test_ms_to_us_and_elapsed_us():
    assert _ms_to_us(1.5) == 1500
    assert _ms_to_us(None) == 0
    assert _ms_to_us("") == 0
    # camel 毫秒键优先 → 转微秒
    assert _elapsed_us({"avgElapsedTime": 2}, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 2000
    # 已是微秒的键透传
    assert _elapsed_us({"avg_elapsed_us": 3000}, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 3000
    assert _elapsed_us({}, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 0


def test_first_present_skips_empty():
    assert _first_present({"a": "", "b": "x"}, ("a", "b")) == "x"
    assert _first_present({}, ("a",), default="-") == "-"


def test_normalize_mode():
    assert _normalize_mode(" MySQL ") == "mysql"
    assert _normalize_mode("oracle") == "oracle"
    with pytest.raises(OcpClientError):
        _normalize_mode(None)


def test_real_sql_rejects_write_before_connect():
    # 只读守卫先于建连：不配 host 也应因写 SQL 直接报错，而不是连接错误
    ex = MysqlSqlExecutor(SqlConfig(provider="real", username="ro", password="pw"))
    with pytest.raises(SqlExecutionError):
        ex.query("delete from orders")


def test_real_executors_share_pooled_base():
    # MySQL 与 Oracle 两个真实执行器结构统一：共用 PooledSqlExecutor 骨架
    assert issubclass(MysqlSqlExecutor, PooledSqlExecutor)
    assert issubclass(OracleSqlExecutor, PooledSqlExecutor)
    assert issubclass(PooledSqlExecutor, ABC)


def test_mysql_resolve_config_builds_user_tenant_cluster():
    out = resolve_mysql_config(SqlConfig(username="ro"), "t1", "c1")
    assert out.username == "ro@t1#c1"
    # 已带 @ 的账号原样保留（允许显式指定租户）
    assert resolve_mysql_config(SqlConfig(username="ro@t2"), "t1", "c1").username == "ro@t2"


def test_oracle_resolve_config_uses_tool_db_name_as_service_name():
    # Oracle 模式的 DSN service name 取工具传入的 db_name（配置里已无 service_name 项）
    out = resolve_oracle_config(
        SqlConfig(host="obproxy", port=2883, username="ro", db_name="shop"), "t1"
    )
    assert out.username == "ro@t1"
    assert out.db_name == "shop"
    assert build_oracle_dsn(out) == "obproxy:2883/shop"
    # 账号已带 @ 原样保留（允许显式指定租户）
    explicit = resolve_oracle_config(SqlConfig(username="ro@t2", db_name="shop"), "t1")
    assert explicit.username == "ro@t2" and explicit.db_name == "shop"
    # db_name 缺失时回退租户名（防御性兜底，正常调用不会发生）
    assert resolve_oracle_config(SqlConfig(username="ro"), "t1").db_name == "t1"


def test_oracle_connection_uses_tool_db_name_as_dsn_service_name():
    # 端到端接线：工具传入的 db_name 就是该 Oracle 租户的 DSN service name
    ex = _create_db_connect(
        "t1",
        "c1",
        "shop",
        "ORACLE",
        SqlConfig(
            provider="real",
            host="{'c1': 'obproxy:2883'}",
            username="ro",
            password="pw",
        ),
    )
    assert isinstance(ex, OracleSqlExecutor)
    assert ex._cfg.username == "ro@t1"
    assert build_oracle_dsn(ex._cfg) == "obproxy:2883/shop"


# ---- 工具错误：分类 + 脱敏 + 「查不到」不再是成功 ----------------------------


def test_connection_error_is_generic_and_leaks_nothing():
    exc = SqlExecutionError(
        "SQL 执行失败: (2003, \"Can't connect to MySQL server on '10.1.2.3' (111)\")"
    )
    kind, message = _classify_error(exc)
    assert kind == "connection"
    assert "10.1.2.3" not in message and "connect" not in message.lower()


def test_sql_error_keeps_semantics_but_redacts_env_details():
    exc = SqlExecutionError(
        "SQL 执行失败: (1054, \"Unknown column 'x' in 'field list' "
        "(ro@t1#c1@10.1.2.3:3306 password=hunter2)\")"
    )
    kind, message = _classify_error(exc)
    assert kind == "sql"
    assert "Unknown column 'x'" in message  # 模型靠它换写法自纠，必须保留
    for secret in ("10.1.2.3", "ro@t1#c1", "hunter2"):
        assert secret not in message


def test_error_payload_hides_raw_message_and_carries_error_id():
    payload = json.loads(_error(RuntimeError("boom-内部细节")))
    assert payload["ok"] is False
    assert payload["error_kind"] == "internal"
    assert "boom-内部细节" not in payload["error"]
    assert "错误编号" in payload["error"]


def test_read_only_violation_keeps_actionable_message():
    kind, message = _classify_error(ReadOnlyViolation("只读模式禁止语句类型: delete"))
    assert kind == "read_only" and "delete" in message


def test_execute_sql_write_reports_read_only_kind(tools):
    data = json.loads(
        tools["execute_sql"].invoke({**_EXEC_ARGS, "sql": "delete from orders"})
    )
    assert data["ok"] is False and data["error_kind"] == "read_only"


class _EmptyPlanOcp(MockOcpClient):
    """计划接口返回空：验证「查不到」必须是 ok:false（旧实现返回裸文本，被审计记成成功）。"""

    def get_sql_top_plan(self, *args, **kwargs):
        return []

    def get_sql_explain(self, *args, **kwargs):
        return []


def test_empty_plan_is_reported_as_failure():
    built = {t.name: t for t in build_tools(_EmptyPlanOcp(), SqlConfig(provider="mock"))}
    top = json.loads(
        built["get_sql_top_plan"].invoke({"cluster_id": 1, "tenant_id": 1001, "sql_id": "sq-1"})
    )
    assert top["ok"] is False and top["error_kind"] == "not_found"
    explain = json.loads(
        built["get_sql_explain"].invoke({"cluster_id": 1, "tenant_id": 1001, "uid": "u-1"})
    )
    assert explain["ok"] is False and explain["error_kind"] == "not_found"