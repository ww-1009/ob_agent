from app.tools.base import (
    ClusterInfo,
    OcpClient,
    QueryResult,
    SqlExecutionError,
    SqlExecutor,
    SlowSqlItem,
    TenantInfo,
    TopologyInfo,
)


def test_slow_sql_item_json_roundtrip():
    item = SlowSqlItem(
        sql_id="abc123",
        sql_text="select 1",
        db_name="shop",
        user_name="u1",
        avg_elapsed_us=1000,
        max_elapsed_us=2000,
        exec_count=10,
        first_seen="2026-08-01T00:00:00",
        last_seen="2026-08-01T01:00:00",
    )
    data = item.model_dump()
    assert data["sql_id"] == "abc123"
    assert SlowSqlItem.model_validate(data) == item


def test_topology_nesting():
    t = TopologyInfo(
        clusters=[
            ClusterInfo(
                cluster_id="c1",
                cluster_name="obcluster-1",
                tenants=[
                    TenantInfo(
                        tenant_id="1001", name="tpcc_mysql", mode="mysql",
                        cluster_id="c1", cluster_name="obcluster-1", status="RUNNING",
                    )
                ],
            )
        ]
    )
    assert t.clusters[0].tenants[0].mode == "mysql"


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
