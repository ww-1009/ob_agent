import pytest

from app.tools.base import SqlExecutionError
from app.tools.sql.mock import MockSqlExecutor


@pytest.fixture
def exe() -> MockSqlExecutor:
    # data 目录相对 backend/ 定位：MockSqlExecutor 默认从 repo backend/data 加载
    return MockSqlExecutor()


def test_select_all_orders(exe):
    r = exe.query("select * from orders")
    assert r.columns[0] == "id"
    assert len(r.rows) == 7
    assert r.truncated is False


def test_select_columns_and_where(exe):
    r = exe.query("select name from customer where id = 42")
    assert r.columns == ["name"]
    assert r.rows == [["Bob"]]


def test_limit_applied(exe):
    r = exe.query("select * from orders limit 2")
    assert len(r.rows) == 2
    assert r.truncated is False


def test_count_star(exe):
    r = exe.query("select count(*) from orders")
    assert r.rows == [[7]]


def test_explain_orders_returns_plan(exe):
    r = exe.explain("explain select * from orders where status = 'PAID'")
    assert r.columns[1] == "OPERATOR"
    assert any("TABLE SCAN" in str(row[1]) for row in r.rows)


def test_gv_sql_audit_returns_slow_rows(exe):
    r = exe.query("select sql_id, avg_elapsed_us from oceanbase.gv$sql_audit limit 10")
    assert len(r.rows) >= 1
    assert "sql_id" in r.columns


def test_unknown_table_raises(exe):
    with pytest.raises(SqlExecutionError):
        exe.query("select * from not_a_table")


def test_unsupported_where_operator_raises(exe):
    with pytest.raises(SqlExecutionError):
        exe.query("select * from orders where total_amount > 100")


def test_multi_condition_where_raises(exe):
    with pytest.raises(SqlExecutionError):
        exe.query("select * from orders where status = 'PAID' and customer_id = 42")


def test_write_sql_rejected(exe):
    with pytest.raises(SqlExecutionError):
        exe.query("update orders set status = 'x'")
