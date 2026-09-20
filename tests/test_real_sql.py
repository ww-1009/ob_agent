import pytest

from app.config import SqlConfig
from app.tools.base import SqlExecutionError
from app.tools.sql.real import RealSqlExecutor, _parse_dsn


def _executor() -> RealSqlExecutor:
    return RealSqlExecutor(
        SqlConfig(
            provider="real",
            dsn="mysql+pymysql://agent_ro:pw@127.0.0.1:3306/shop",
            connect_timeout=1,
            query_timeout_seconds=1,
        )
    )


def test_write_sql_rejected_before_connect():
    exe = _executor()
    with pytest.raises(SqlExecutionError, match="只读|禁止"):
        exe.query("delete from orders")


def test_missing_dsn_raises_clear_error():
    exe = RealSqlExecutor(SqlConfig(provider="real", dsn="", username="", password=""))
    with pytest.raises(SqlExecutionError, match="dsn|未配置"):
        exe.query("select 1")


def test_parse_dsn_plain():
    d = _parse_dsn("mysql+pymysql://agent_ro:pw@127.0.0.1:3306/shop")
    assert d == {"user": "agent_ro", "password": "pw", "host": "127.0.0.1", "port": 3306, "db": "shop"}


def test_parse_dsn_password_with_at():
    d = _parse_dsn("mysql+pymysql://u:p@ss@h:3306/db")
    assert d["password"] == "p@ss"
    assert d["host"] == "h"


def test_parse_dsn_password_percent_encoded():
    d = _parse_dsn("mysql+pymysql://u:p%40ss@h:3306/db")
    assert d["password"] == "p@ss"


def test_parse_dsn_malformed_scheme_raises():
    with pytest.raises(SqlExecutionError):
        _parse_dsn("postgres://u:p@h:3306/db")


def test_parse_dsn_bad_port_raises():
    with pytest.raises(SqlExecutionError):
        _parse_dsn("mysql+pymysql://u:p@h:abc/db")
