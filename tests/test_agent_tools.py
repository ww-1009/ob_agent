"""agent 工具层测试：验证 4 个工具的行为与观察结果契约（spec §8.1）。

契约：工具内部异常一律转成 {"ok": false, "error": ...} 的 JSON 字符串；
成功返回 {"ok": true, ...} 的 JSON 字符串。query_db / explain_sql 执行前先过 assert_read_only。
"""
import json

import pytest

from app.agent.tools import build_tools
from app.tools.base import QueryResult
from app.tools.ocp.mock import MockOcpClient
from app.tools.sql.mock import MockSqlExecutor


class _NoGuardSqlExecutor:
    """无守卫桩：不跑 assert_read_only，用于验证写 SQL 在工具层就被挡（未到达执行器）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def query(self, sql: str) -> QueryResult:
        self.calls.append(sql)
        return QueryResult(columns=["c"], rows=[[1]], row_count=1, truncated=False)

    def explain(self, sql: str) -> QueryResult:
        self.calls.append(sql)
        return QueryResult(columns=["OPERATOR"], rows=[["TABLE SCAN"]], row_count=1, truncated=False)


@pytest.fixture
def tools():
    return build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)


def _tool_by_name(tools, name):
    return next(t for t in tools if t.name == name)


def test_get_topology(tools):
    out = _tool_by_name(tools, "get_topology").invoke({})
    data = json.loads(out)
    assert data["ok"] is True
    names = [t["name"] for c in data["clusters"] for t in c["tenants"]]
    assert "tpcc_mysql" in names


def test_query_slow_sql(tools):
    out = _tool_by_name(tools, "query_slow_sql").invoke(
        {"top_n": 2, "tenant_id": "1001"}
    )
    data = json.loads(out)
    assert data["ok"] is True
    items = data["items"]
    assert len(items) == 2
    assert items[0]["sql_id"]


def test_query_db_returns_rows(tools):
    out = _tool_by_name(tools, "query_db").invoke({"sql": "select * from customer"})
    data = json.loads(out)
    assert data["ok"] is True
    assert data["columns"] == ["id", "name", "level"]
    assert len(data["rows"]) == 3


def test_query_db_write_is_error_observation(tools):
    out = _tool_by_name(tools, "query_db").invoke({"sql": "drop table orders"})
    data = json.loads(out)
    assert data["ok"] is False
    assert "error" in data


def test_send_row_data_false_strips_rows():
    sql = MockSqlExecutor()
    tools = build_tools(MockOcpClient(), sql, send_row_data=False)
    out = _tool_by_name(tools, "query_db").invoke({"sql": "select * from customer"})
    data = json.loads(out)
    assert data["ok"] is True
    assert data["rows"] == []
    assert data["row_count"] == 3


def test_explain_sql(tools):
    out = _tool_by_name(tools, "explain_sql").invoke(
        {"sql": "select * from orders where status='PAID'"}
    )
    data = json.loads(out)
    assert data["ok"] is True
    assert "OPERATOR" in data["columns"]


def test_query_db_write_blocked_by_tool_guard_before_executor():
    ex = _NoGuardSqlExecutor()
    tools = build_tools(MockOcpClient(), ex, send_row_data=True)
    out = _tool_by_name(tools, "query_db").invoke({"sql": "drop table orders"})
    data = json.loads(out)
    assert data["ok"] is False and "error" in data
    assert ex.calls == []  # 写 SQL 未到达执行器 → 工具层守卫生效


def test_query_db_read_reaches_executor():
    ex = _NoGuardSqlExecutor()
    tools = build_tools(MockOcpClient(), ex, send_row_data=True)
    out = _tool_by_name(tools, "query_db").invoke({"sql": "select * from customer"})
    assert json.loads(out)["ok"] is True
    assert ex.calls == ["select * from customer"]


def test_first_mysql_tenant():
    from app.agent.tools import _first_mysql_tenant

    assert _first_mysql_tenant(MockOcpClient()) == "1001"  # fixture: mysql 1001 + oracle 1002


def test_query_slow_sql_forwards_resolved_tenant():
    class _RecordingOcp(MockOcpClient):
        def __init__(self):
            super().__init__()
            self.seen: list[tuple[str, int]] = []

        def list_slow_sql(self, tenant_id: str, top_n: int = 10):
            self.seen.append((tenant_id, top_n))
            return super().list_slow_sql(tenant_id, top_n)

    ocp = _RecordingOcp()
    tools = build_tools(ocp, MockSqlExecutor(), send_row_data=True)
    out = _tool_by_name(tools, "query_slow_sql").invoke({"top_n": 2})
    data = json.loads(out)
    assert data["ok"] is True
    assert ocp.seen == [("1001", 2)]


def test_query_db_executor_error_is_observation():
    tools = build_tools(MockOcpClient(), MockSqlExecutor(), send_row_data=True)
    out = _tool_by_name(tools, "query_db").invoke({"sql": "select * from not_a_table"})
    data = json.loads(out)
    assert data["ok"] is False and "error" in data


def test_get_topology_ocp_error_is_observation():
    class _BrokenOcp(MockOcpClient):
        def get_topology(self):
            raise RuntimeError("ocp down")

    tools = build_tools(_BrokenOcp(), MockSqlExecutor(), send_row_data=True)
    out = _tool_by_name(tools, "get_topology").invoke({})
    data = json.loads(out)
    assert data["ok"] is False and "error" in data
