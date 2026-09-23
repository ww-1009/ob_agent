from app.tools.base import OcpClient, QueryResult, SqlExecutionError, SqlExecutor


def test_query_result_defaults_truncated_false():
    qr = QueryResult(columns=["a"], rows=[[1]])
    assert qr.truncated is False
    assert qr.row_count == 1


def test_query_result_explicit_row_count_len_wins():
    qr = QueryResult(columns=["a"], rows=[[1], [2]], row_count=99)
    assert qr.row_count == 2


def test_query_result_json_roundtrip_keeps_row_count_in_sync():
    qr = QueryResult(columns=["a"], rows=[[1], [2]], truncated=True)
    restored = QueryResult.model_validate_json(qr.model_dump_json())
    assert restored.row_count == 2
    assert restored.truncated is True
    assert restored.rows == [[1], [2]]


def test_sql_execution_error_is_value_error():
    assert issubclass(SqlExecutionError, ValueError)


def test_protocols_are_runtime_checkable():
    from app.tools.ocp.mock import MockOcpClient
    from app.tools.sql.mock import MockSqlExecutor

    assert isinstance(MockOcpClient(), OcpClient)
    assert isinstance(MockSqlExecutor(), SqlExecutor)
    assert not isinstance(object(), SqlExecutor)
