import pytest

from app.tools.ocp.mock import MockOcpClient


@pytest.fixture
def client() -> MockOcpClient:
    return MockOcpClient()


def test_get_topology_has_mysql_tenant(client):
    topo = client.get_topology()
    names = {t.name for c in topo.clusters for t in c.tenants}
    assert "tpcc_mysql" in names
    mysql_modes = {t.mode for c in topo.clusters for t in c.tenants if t.name == "tpcc_mysql"}
    assert mysql_modes == {"mysql"}


def test_list_slow_sql_returns_items(client):
    items = client.list_slow_sql(tenant_id="1001", top_n=1)
    assert len(items) == 1
    assert items[0].sql_id == "sq-scan-orders-1"
    assert items[0].avg_elapsed_us > 0


def test_list_slow_sql_respects_top_n(client):
    items = client.list_slow_sql(tenant_id="1001", top_n=5)
    assert len(items) == 2  # fixtures 只有 2 条
