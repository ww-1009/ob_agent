"""精简后的核心通路测试：只覆盖当前 API 的关键行为。

替代已删除的 test_agent_tools.py / test_real_ocp.py / test_real_sql.py 中仍有效的那部分
契约：工具注册名、工具层只读守卫、real 层纯函数、MysqlSqlExecutor 连接前拒绝写 SQL，
以及 MySQL / Oracle 两个真实执行器的命名与结构对称性。
"""
import json
from abc import ABC

import pytest

from app.agent.tools import build_tools
from app.config import SqlConfig
from app.tools.base import OcpClientError, SqlExecutionError
from app.tools.ocp.mock import MockOcpClient
from app.tools.ocp.real import _elapsed_us, _first_present, _ms_to_us, _normalize_mode
from app.tools.sql.base import PooledSqlExecutor
from app.tools.sql.oracle import OracleSqlExecutor, resolve_config as resolve_oracle_config
from app.tools.sql.real import MysqlSqlExecutor, resolve_config as resolve_mysql_config

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


def test_oracle_resolve_config_fills_tenant_and_service_name():
    out = resolve_oracle_config(SqlConfig(username="ro"), "t1")
    assert out.username == "ro@t1"
    assert out.service_name == "t1"
    # 显式配置优先
    explicit = resolve_oracle_config(
        SqlConfig(username="ro@t2", service_name="t1#c1"), "t1"
    )
    assert explicit.username == "ro@t2" and explicit.service_name == "t1#c1"