import pytest

from app.tools.ocp.mock import MockOcpClient


@pytest.fixture
def client() -> MockOcpClient:
    return MockOcpClient()


def test_get_tenant_info_matches_by_id(client):
    assert client.get_tenant_info(1, "1001")["name"] == "tpcc_mysql"
    # 查不到返回空 dict（工具层据此报 not_found），不抛异常
    assert client.get_tenant_info(1, "does-not-exist") == {}


def test_get_slow_sql_reads_fixture_and_honours_limit(client):
    items = client.get_slow_sql(
        1, "1001", "2026-01-01T00:00:00", "2026-01-01T01:00:00", limit=1
    )
    assert [it["sqlId"] for it in items] == ["sq-scan-orders-1"]
    assert items[0]["avgElapsedTime"] > 0