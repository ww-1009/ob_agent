import json
from pathlib import Path

import pytest

from app.tools.ocp.mock import MockOcpClient, _EXPLAIN_BY_UID

_FIXTURE = Path(__file__).resolve().parents[1] / "backend" / "data" / "ocp_slow_sqls.json"


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
    assert len(items) == 1  # 遵守 limit
    # 断言来自夹具本身，而不是写死某个 sqlId：夹具换数据时这条用例不该跟着过期
    fixture_ids = {
        row["sqlId"] for row in json.loads(_FIXTURE.read_text(encoding="utf-8"))
    }
    assert items[0]["sqlId"] in fixture_ids
    assert items[0]["avgElapsedTime"] > 0


def test_cluster_resource_stats_follow_cluster_id(client):
    stats = client.get_cluster_resource_stats(1)
    assert stats["clusterName"] == "obcluster"
    assert stats["cpuTotal"] > 0
    # 集群不存在返回空对象，工具层据此报 not_found（而不是把空值当成零水位）
    assert client.get_cluster_resource_stats(999) == {}


def test_server_resource_stats_follow_cluster_id(client):
    servers = client.get_server_resource_stats(1)
    assert len(servers) == 3
    assert {s["zone"] for s in servers} == {"zone1", "zone2", "zone3"}
    assert client.get_server_resource_stats(999) == []


def test_explain_fixtures_cover_every_top_plan_uid(client):
    """compare_plans 靠 uid → 夹具映射区分「改动前 / 改动后」两份计划。

    夹具是生成的（top plan 的 uid 是 base64 串），重生成后映射不会自动跟着变；
    这条用例让漏更新立刻失败，而不是静默退回成「两份计划一模一样」。
    """
    items = client.get_sql_top_plan(1, 1001, "sq-1", "2026-01-01T00:00:00", "2026-01-01T01:00:00")
    uids = [item["uid"] for item in items]
    assert uids, "top plan 夹具必须给出计划 uid"
    assert set(uids) <= set(_EXPLAIN_BY_UID)
    fixtures = [_EXPLAIN_BY_UID[uid] for uid in uids]
    assert len(set(fixtures)) == len(fixtures), "两个 uid 必须对应两份不同的计划夹具"
    for uid in uids:
        assert client.get_sql_explain(1, 1001, uid, "2026-01-01T00:00:00", "2026-01-01T01:00:00")