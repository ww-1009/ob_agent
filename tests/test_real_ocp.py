import base64
import re
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.config import OcpConfig
from app.tools.base import OcpClientError, SlowSqlItem
from app.tools.ocp.real import (
    RealOcpClient,
    _DEFAULT_SLOW_SQL_WINDOW_MINUTES,
    _elapsed_us,
    _first_present,
    _iso_utc,
    _ms_to_us,
    _normalize_mode,
    map_cluster,
    map_slow_sql,
    map_tenant,
    map_topology,
    slow_sql_time_window,
)


# ---------- map_slow_sql：单位语义（camel 毫秒 → 微秒） ----------


def test_map_slow_sql_camelcase_ms_to_us():
    raw = {
        "sqlId": "abc",
        "fulltext": "select * from orders",
        "dbName": "shop",
        "userName": "u1",
        "avgElapsedTime": 1.2,  # 单位毫秒（OCP 真实报文）
        "maxElapsedTime": 3.5,  # 单位毫秒
        "executions": 100,
    }
    item = map_slow_sql(raw)
    assert isinstance(item, SlowSqlItem)
    assert item.sql_id == "abc"
    assert item.avg_elapsed_us == 1200  # ms → us
    assert item.max_elapsed_us == 3500


def test_map_slow_sql_handles_missing_optional():
    item = map_slow_sql({"sqlId": "x"})
    assert item.sql_text == ""
    assert item.max_elapsed_us == 0
    assert item.first_seen is None
    assert item.last_seen is None


def test_map_slow_sql_snake_case_us_keys():
    raw = {
        "sql_id": "s1",
        "sql_text": "select 1",
        "db_name": "db1",
        "user_name": "u1",
        "avg_elapsed_us": 3000,  # 已是微秒，透传
        "max_elapsed_us": 4500,
        "exec_count": 12,
        "first_time": "2026-01-01 10:00:00",
        "last_time": "2026-01-02 11:00:00",
    }
    item = map_slow_sql(raw)
    assert item.sql_id == "s1"
    assert item.sql_text == "select 1"
    assert item.db_name == "db1"
    assert item.user_name == "u1"
    assert item.avg_elapsed_us == 3000
    assert item.max_elapsed_us == 4500
    assert item.exec_count == 12
    assert item.first_seen == "2026-01-01 10:00:00"
    assert item.last_seen == "2026-01-02 11:00:00"


def test_map_slow_sql_snake_ms_fallback_keys():
    raw = {
        "sql_id": "s2",
        "sql_text": "select 2",
        "avg_elapsed_ms": 3.0,
        "max_elapsed_ms": 4.0,
        "exec_count": 2,
    }
    item = map_slow_sql(raw)
    assert item.avg_elapsed_us == 3000
    assert item.max_elapsed_us == 4000


def test_map_slow_sql_camel_ms_wins_when_both_present():
    item = map_slow_sql(
        {"sqlId": "a", "sql_id": "b", "avgElapsedTime": 2.0, "avg_elapsed_ms": 99}
    )
    assert item.sql_id == "a"
    assert item.avg_elapsed_us == 2000


def test_map_slow_sql_camel_exec_and_timestamps():
    item = map_slow_sql(
        {
            "sqlId": "x",
            "executions": 7,
            "firstTime": "2026-01-01 08:00:00",
            "lastTime": "2026-01-02 09:00:00",
        }
    )
    assert item.exec_count == 7
    assert item.first_seen == "2026-01-01 08:00:00"
    assert item.last_seen == "2026-01-02 09:00:00"


def test_map_slow_sql_prefers_fulltext_over_short():
    assert (
        map_slow_sql(
            {"sqlId": "s", "fulltext": "FULL", "sqlTextShort": "SHORT"}
        ).sql_text
        == "FULL"
    )
    # 完整文本键缺省时才回落 short 键
    assert map_slow_sql({"sqlId": "s", "sqlTextShort": "SHORT"}).sql_text == "SHORT"
    # camel short 优先于 snake short
    assert (
        map_slow_sql(
            {"sqlId": "s", "sql_text_short": "SSHORT", "sqlTextShort": "CSHORT"}
        ).sql_text
        == "CSHORT"
    )


def test_map_slow_sql_none_values_default():
    item = map_slow_sql({"sqlId": None, "fulltext": None, "avgElapsedTime": None})
    assert item.sql_id == ""
    assert item.sql_text == ""
    assert item.avg_elapsed_us == 0
    assert item.exec_count == 0


# ---------- map_slow_sql：A/B 键容忍锚点 ----------


def test_map_slow_sql_ab_key_tolerance_equivalence():
    a = {
        "sqlId": "s1",
        "fulltext": "select * from orders",
        "dbName": "shop",
        "userName": "u1",
        "avgElapsedTime": 1.2,
        "maxElapsedTime": 3.5,
        "executions": 100,
        "firstTime": "2026-01-01 08:00:00",
        "lastTime": "2026-01-02 09:00:00",
    }
    b = {
        "sql_id": "s1",
        "sql_text": "select * from orders",
        "db_name": "shop",
        "user_name": "u1",
        "avg_elapsed_ms": 1.2,
        "max_elapsed_ms": 3.5,
        "exec_count": 100,
        "first_time": "2026-01-01 08:00:00",
        "last_time": "2026-01-02 09:00:00",
    }
    assert map_slow_sql(a) == map_slow_sql(b)


def test_map_slow_sql_snake_first_seen_last_seen_keys():
    # snake 侧 fixture 键 first_seen/last_seen（共享模型字段名 + data fixture 键）同样能映射，
    # 且与 camel firstTime/lastTime 产出字段全等
    snake = map_slow_sql(
        {
            "sql_id": "s1",
            "sql_text": "select 1",
            "avg_elapsed_us": 1000,
            "first_seen": "2026-01-01 10:00:00",
            "last_seen": "2026-01-02 11:00:00",
        }
    )
    camel = map_slow_sql(
        {
            "sqlId": "s1",
            "fulltext": "select 1",
            "avgElapsedTime": 1.0,
            "firstTime": "2026-01-01 10:00:00",
            "lastTime": "2026-01-02 11:00:00",
        }
    )
    assert snake.first_seen == "2026-01-01 10:00:00"
    assert snake.last_seen == "2026-01-02 11:00:00"
    assert snake == camel


# ---------- 纯辅助函数 ----------


def test_first_present_returns_first_nonempty_in_order():
    assert _first_present({"a": "x", "b": "y"}, ("a", "b")) == "x"


def test_first_present_skips_none_and_empty_then_default():
    assert _first_present({"a": None, "b": "", "c": 5}, ("a", "b", "c")) == "5"
    assert _first_present({"a": None, "b": ""}, ("a", "b"), default="d") == "d"
    assert _first_present({}, ("a",), default="z") == "z"


def test_first_present_coerces_non_str_to_str():
    assert _first_present({"a": 123}, ("a",)) == "123"


def test_ms_to_us():
    assert _ms_to_us(1.2) == 1200
    assert _ms_to_us("3.5") == 3500
    assert _ms_to_us(0) == 0
    assert _ms_to_us(None) == 0
    assert _ms_to_us("") == 0
    assert _ms_to_us("   ") == 0


def test_elapsed_us_camel_ms_priority():
    raw = {"avgElapsedTime": 1.2, "avg_elapsed_us": 9999, "avg_elapsed_ms": 88}
    assert _elapsed_us(raw, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 1200


def test_elapsed_us_us_passthrough():
    raw = {"avg_elapsed_us": 3000}
    assert _elapsed_us(raw, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 3000


def test_elapsed_us_ms_fallback():
    raw = {"avg_elapsed_ms": 3.0}
    assert _elapsed_us(raw, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 3000


def test_elapsed_us_all_missing_zero():
    assert _elapsed_us({}, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 0


def test_elapsed_us_none_handled():
    raw = {"avgElapsedTime": None, "avg_elapsed_us": None, "avg_elapsed_ms": None}
    assert _elapsed_us(raw, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 0


def test_elapsed_us_rounds_non_integer():
    raw = {"avgElapsedTime": 0.9999}
    assert _elapsed_us(raw, "avgElapsedTime", "avg_elapsed_us", "avg_elapsed_ms") == 1000


def test_normalize_mode_normalizes_case_and_whitespace():
    assert _normalize_mode("MYSQL") == "mysql"
    assert _normalize_mode("mysql") == "mysql"
    assert _normalize_mode(" mysql ") == "mysql"
    assert _normalize_mode("ORACLE") == "oracle"
    assert _normalize_mode("Oracle") == "oracle"


def test_normalize_mode_unknown_raises():
    with pytest.raises(OcpClientError, match="mode"):
        _normalize_mode("postgres")


def test_normalize_mode_none_raises():
    with pytest.raises(OcpClientError, match="mode"):
        _normalize_mode(None)


def test_iso_utc_z_format():
    dt = datetime(2026, 9, 9, 1, 2, 3, tzinfo=timezone.utc)
    assert _iso_utc(dt) == "2026-09-09T01:02:03Z"


def test_slow_sql_time_window_deterministic():
    now = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    start, end = slow_sql_time_window(now)
    assert end == "2026-09-09T12:00:00Z"
    assert start == "2026-09-09T11:30:00Z"
    # 格式符合 %Y-%m-%dT%H:%M:%SZ 且跨度恰为默认 30 分钟
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    assert pat.match(start)
    assert pat.match(end)
    start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert end_dt - start_dt == timedelta(minutes=_DEFAULT_SLOW_SQL_WINDOW_MINUTES)
    assert end_dt == now


# ---------- map_tenant / map_cluster ----------


def test_map_tenant_name_priority_and_defaults():
    t = map_tenant(
        {
            "id": "9",
            "name": "app",
            "tenantName": "ignored",
            "mode": " mysql ",
            "obClusterId": "c1",
            "clusterName": "ob1",
            "status": "ACTIVE",
        }
    )
    assert t.tenant_id == "9"
    assert t.name == "app"
    assert t.mode == "mysql"
    assert t.cluster_id == "c1"
    assert t.cluster_name == "ob1"
    assert t.status == "ACTIVE"


def test_map_tenant_uses_cluster_defaults_when_keys_missing():
    t = map_tenant(
        {"id": 1, "tenantName": "app", "mode": "MYSQL"},
        default_cluster_id="9",
        default_cluster_name="c9",
    )
    assert t.cluster_id == "9"
    assert t.cluster_name == "c9"
    assert t.status == "RUNNING"


def test_map_tenant_missing_mode_raises():
    with pytest.raises(OcpClientError, match="mode"):
        map_tenant({"id": "1", "name": "app1"})


def test_map_cluster_camel_and_snake():
    c1 = map_cluster({"id": 2, "name": "ob-2"})
    assert c1.cluster_id == "2"
    assert c1.cluster_name == "ob-2"
    assert c1.tenants == []
    c2 = map_cluster({"cluster_id": 2, "cluster_name": "ob-2"})
    assert c2.cluster_id == "2"
    assert c2.cluster_name == "ob-2"


# ---------- map_topology ----------


def test_map_topology_basic_structure():
    topo = map_topology(
        [{"id": 1, "name": "ob-a"}],
        [{"id": 1001, "tenantName": "app1", "mode": "MYSQL", "obClusterId": 1}],
    )
    assert len(topo.clusters) == 1
    c = topo.clusters[0]
    assert c.cluster_id == "1"
    assert c.cluster_name == "ob-a"
    assert len(c.tenants) == 1
    t = c.tenants[0]
    assert t.tenant_id == "1001"
    assert t.name == "app1"
    assert t.mode == "mysql"
    assert t.cluster_id == "1"
    assert t.cluster_name == "ob-a"
    assert t.status == "RUNNING"


def test_map_topology_ab_key_tolerance_equivalence():
    clusters_a = [
        {"id": 1, "name": "ob-a"},
        {"id": 2, "name": "ob-b"},
    ]
    tenants_a = [
        {"id": 101, "tenantName": "app1", "mode": "MYSQL", "obClusterId": 1, "clusterName": "ob-a", "status": "RUNNING"},
        {"id": 102, "tenantName": "app2", "mode": "ORACLE", "obClusterId": 2, "clusterName": "ob-b"},
        {"id": 103, "tenantName": "ocpmonitor", "mode": "MYSQL", "obClusterId": 1},  # 系统租户，过滤
        {"id": 104, "tenantName": "sys", "mode": "MYSQL", "obClusterId": 1},  # 系统租户，过滤
    ]
    clusters_b = [
        {"cluster_id": 1, "cluster_name": "ob-a"},
        {"cluster_id": 2, "cluster_name": "ob-b"},
    ]
    tenants_b = [
        {"tenant_id": 101, "name": "app1", "mode_name": "mysql", "cluster_id": 1, "cluster_name": "ob-a"},
        {"tenant_id": 102, "name": "app2", "ob_compatibility_mode": "oracle", "cluster_id": 2, "cluster_name": "ob-b"},
        {"tenant_id": 103, "name": "OCPMonitor", "mode": "mysql", "cluster_id": 1},
        {"tenant_id": 104, "name": "Sys", "mode": "mysql", "cluster_id": 1},
    ]
    topo_a = map_topology(clusters_a, tenants_a)
    topo_b = map_topology(clusters_b, tenants_b)
    assert topo_a == topo_b
    assert topo_a.model_dump() == topo_b.model_dump()
    assert [c.cluster_id for c in topo_a.clusters] == ["1", "2"]
    assert {
        c.cluster_id: [t.name for t in c.tenants] for c in topo_a.clusters
    } == {"1": ["app1"], "2": ["app2"]}


def test_map_topology_cluster_id_variants_not_orphaned():
    # 集群条目自身 id 以 clusterId/obClusterId 形式出现，map_cluster 也应认到，
    # 使租户能正确归并（不产生孤儿租户 / 重复合成集群）
    topo = map_topology(
        [
            {"clusterId": 11, "name": "ob-a"},
            {"obClusterId": 12, "clusterName": "ob-b"},
        ],
        [
            {"id": 1, "tenantName": "app-a", "mode": "MYSQL", "clusterId": 11},
            {"id": 2, "tenantName": "app-b", "mode": "MYSQL", "obClusterId": 12},
        ],
    )
    assert [c.cluster_id for c in topo.clusters] == ["11", "12"]
    assert {
        c.cluster_id: [t.name for t in c.tenants] for c in topo.clusters
    } == {"11": ["app-a"], "12": ["app-b"]}


@pytest.mark.parametrize(
    "sys_name",
    ["sys", "Sys", "SYS", "ocpmeta", "OCPMETA", "ocpmonitor", "OcpMonitor"],
)
def test_map_topology_filters_system_tenants(sys_name):
    topo = map_topology(
        [{"id": 1, "name": "ob-a"}],
        [
            {"id": 1001, "name": "app1", "mode": "MYSQL", "obClusterId": 1},
            {"id": 1002, "name": sys_name, "mode": "MYSQL", "obClusterId": 1},
        ],
    )
    assert len(topo.clusters) == 1
    assert [t.name for t in topo.clusters[0].tenants] == ["app1"]


def test_map_topology_synthetic_cluster_keeps_tenant():
    topo = map_topology(
        [{"id": 1, "name": "c1"}],
        [
            # 引用了 clusters 列表里不存在的 obClusterId → 需合成集群殿后
            {"id": 10, "tenantName": "t1", "mode": "MYSQL", "obClusterId": 2, "clusterName": "c2"},
            {"id": 11, "tenantName": "t2", "mode": "MYSQL", "obClusterId": 1},
        ],
    )
    assert [c.cluster_id for c in topo.clusters] == ["1", "2"]
    by_id = {c.cluster_id: c for c in topo.clusters}
    assert by_id["1"].cluster_name == "c1"
    assert [t.name for t in by_id["1"].tenants] == ["t2"]
    assert by_id["2"].cluster_name == "c2"
    assert [t.name for t in by_id["2"].tenants] == ["t1"]
    # 租户归属合成集群的 id/name 被作为 default 透传
    tenant = by_id["2"].tenants[0]
    assert tenant.cluster_id == "2"
    assert tenant.cluster_name == "c2"


def test_map_topology_unknown_mode_raises():
    with pytest.raises(OcpClientError, match="mode"):
        map_topology(
            [{"id": 1, "name": "c1"}],
            [{"id": 1, "name": "app1", "mode": "postgres", "obClusterId": 1}],
        )


# ---------- RealOcpClient 骨架保留 ----------


@pytest.mark.parametrize(
    "invoke",
    [
        lambda c: c.get_topology(),
        lambda c: c.list_slow_sql(tenant_id="1001"),
    ],
    ids=["get_topology", "list_slow_sql"],
)
def test_missing_config_raises_clear_error(invoke):
    client = RealOcpClient(OcpConfig(provider="real", base_url=""))
    with pytest.raises(OcpClientError, match="base_url"):
        invoke(client)


def test_ocp_client_error_is_value_error():
    assert issubclass(OcpClientError, ValueError)


# ---------- RealOcpClient：HTTP 层（httpx.MockTransport） ----------


_BASIC_AUTH = "Basic " + base64.b64encode(b"ocp:secret").decode()

_CLUSTERS_RAW = [{"id": 1, "name": "ob-a"}]
_TENANTS_RAW = [
    {"id": 1001, "name": "app1", "mode": "MYSQL", "obClusterId": 1, "status": "RUNNING"},
    {"id": 1002, "name": "app2", "mode": "ORACLE", "obClusterId": 1},
    # 系统租户：大小写不敏感，均应在映射阶段被过滤
    {"id": 9001, "name": "sys", "mode": "MYSQL", "obClusterId": 1},
    {"id": 9002, "name": "OCPmeta", "mode": "MYSQL", "obClusterId": 1},
    {"id": 9003, "name": "ocpmonitor", "mode": "MYSQL", "obClusterId": 1},
]
_SLOW_SQL_RAW = [
    {
        "sqlId": "abc",
        "fulltext": "select * from orders",
        "dbName": "shop",
        "userName": "u1",
        "avgElapsedTime": 1.2,  # ms
        "maxElapsedTime": 3.5,  # ms
        "executions": 100,
    }
]

_TOPOLOGY_OK = {
    "/api/v2/ob/clusters": _CLUSTERS_RAW,
    "/api/v2/ob/tenants": _TENANTS_RAW,
}


def _cfg(base_url: str = "https://ocp.example.com", **kw) -> OcpConfig:
    defaults = {
        "provider": "real",
        "base_url": base_url,
        "username": "ocp",
        "password": "secret",
        "verify_ssl": True,
    }
    defaults.update(kw)
    return OcpConfig(**defaults)


def _envelope(path: str, contents) -> httpx.Response:
    return httpx.Response(
        200,
        json={"successful": True, "data": {"contents": contents}},
    )


def test_get_topology_success_with_basic_auth():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _envelope(request.url.path, _TOPOLOGY_OK[request.url.path])

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    topo = client.get_topology()

    assert len(seen) == 2
    for req in seen:
        assert req.headers.get("Authorization") == _BASIC_AUTH
    assert [c.cluster_id for c in topo.clusters] == ["1"]
    cluster = topo.clusters[0]
    assert cluster.cluster_name == "ob-a"
    # 系统租户 sys/OCPmeta/ocpmonitor 被过滤，app2 的 ORACLE 大写 mode 被归一
    assert [t.name for t in cluster.tenants] == ["app1", "app2"]
    assert cluster.tenants[0].mode == "mysql"
    assert cluster.tenants[1].mode == "oracle"
    assert cluster.tenants[0].cluster_id == "1"
    assert cluster.tenants[0].cluster_name == "ob-a"
    assert cluster.tenants[0].status == "RUNNING"


def test_get_topology_makes_exactly_two_requests():
    n = {"requests": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        n["requests"] += 1
        return _envelope(request.url.path, _TOPOLOGY_OK[request.url.path])

    topo = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler)).get_topology()

    assert n["requests"] == 2
    assert len(topo.clusters) == 1


def test_envelope_unsuccessful_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"successful": False, "data": {}})

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="unsuccessful"):
        client.get_topology()


def test_http_404_raises_with_status_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="404"):
        client.get_topology()


def test_transport_timeout_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="boom"):
        client.get_topology()


def test_non_json_body_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>")

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="JSON"):
        client.get_topology()


def test_successful_without_data_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"successful": True})

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="data"):
        client.get_topology()


def test_data_without_contents_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"successful": True, "data": {"foo": 1}})

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="contents"):
        client.get_topology()


def test_list_slow_sql_success_params_and_units(monkeypatch):
    monkeypatch.setattr(
        "app.tools.ocp.real.slow_sql_time_window",
        lambda *a, **k: ("2026-09-09T00:00:00Z", "2026-09-09T00:30:00Z"),
    )
    slow_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in _TOPOLOGY_OK:
            return _envelope(path, _TOPOLOGY_OK[path])
        if path == "/api/v2/ob/clusters/1/tenants/1001/slowSql":
            slow_request["request"] = request
            return _envelope(path, _SLOW_SQL_RAW)
        return httpx.Response(404)

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    items = client.list_slow_sql("1001", top_n=5)

    req = slow_request["request"]
    assert req.url.path == "/api/v2/ob/clusters/1/tenants/1001/slowSql"
    params = req.url.params
    assert params["startTime"] == "2026-09-09T00:00:00Z"
    assert params["endTime"] == "2026-09-09T00:30:00Z"
    assert params["limit"] == "5"
    assert params["sqlTextLength"] == "2000"
    assert len(items) == 1
    assert items[0].sql_id == "abc"
    assert items[0].avg_elapsed_us == 1200  # ms → us
    assert items[0].max_elapsed_us == 3500
    assert items[0].exec_count == 100


def test_list_slow_sql_unknown_tenant_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in _TOPOLOGY_OK:
            return _envelope(path, _TOPOLOGY_OK[path])
        return httpx.Response(404)

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="9999"):
        client.list_slow_sql("9999")


def test_get_topology_empty_contents_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        return _envelope(request.url.path, [])

    topo = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler)).get_topology()
    assert topo.clusters == []


def test_list_slow_sql_empty_contents_ok(monkeypatch):
    monkeypatch.setattr(
        "app.tools.ocp.real.slow_sql_time_window",
        lambda *a, **k: ("2026-09-09T00:00:00Z", "2026-09-09T00:30:00Z"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in _TOPOLOGY_OK:
            return _envelope(path, _TOPOLOGY_OK[path])
        if path == "/api/v2/ob/clusters/1/tenants/1001/slowSql":
            return _envelope(path, [])
        return httpx.Response(404)

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    assert client.list_slow_sql("1001", top_n=5) == []


def test_map_unknown_mode_passthrough_ocp_error():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/ob/clusters":
            return _envelope(path, _CLUSTERS_RAW)
        if path == "/api/v2/ob/tenants":
            return _envelope(
                path,
                [{"id": 1001, "name": "app1", "mode": "postgres", "obClusterId": 1}],
            )
        return httpx.Response(404)

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="mode"):
        client.get_topology()


def test_slow_sql_map_valueerror_wrapped_into_ocp_error(monkeypatch):
    monkeypatch.setattr(
        "app.tools.ocp.real.slow_sql_time_window",
        lambda *a, **k: ("2026-09-09T00:00:00Z", "2026-09-09T00:30:00Z"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in _TOPOLOGY_OK:
            return _envelope(path, _TOPOLOGY_OK[path])
        if path == "/api/v2/ob/clusters/1/tenants/1001/slowSql":
            # executions 为列表 → int(...) 抛 TypeError → 应被包成 OcpClientError
            return _envelope(path, [{"sqlId": "x", "executions": [1, 2]}])
        return httpx.Response(404)

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError):
        client.list_slow_sql("1001", top_n=5)


# ---------- RealOcpClient：code-review 硬化（原因保留 / 空认证 / 缓存 / 畸形入参） ----------


def test_unsuccessful_includes_server_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"successful": False, "data": {}, "message": "token expired"},
        )

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="token expired"):
        client.get_topology()


def test_http_error_includes_response_body_snippet():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="no such cluster")

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="no such cluster"):
        client.get_topology()


def test_no_authorization_header_when_username_empty():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _envelope(request.url.path, _TOPOLOGY_OK[request.url.path])

    client = RealOcpClient(_cfg(username=""), transport=httpx.MockTransport(handler))
    topo = client.get_topology()

    assert len(seen) == 2
    for req in seen:
        assert "Authorization" not in req.headers
    assert len(topo.clusters) == 1


def test_client_built_once_with_verify_timeout_passthrough(monkeypatch):
    calls = []
    original_client = httpx.Client

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", spy)

    def handler(request: httpx.Request) -> httpx.Response:
        return _envelope(request.url.path, _TOPOLOGY_OK.get(request.url.path, []))

    client = RealOcpClient(
        _cfg(verify_ssl=False), transport=httpx.MockTransport(handler)
    )
    client.get_topology()
    client.get_topology()

    assert len(calls) == 1  # _client 缓存复用
    assert calls[0]["verify"] is False
    assert calls[0]["timeout"] == 10.0


def test_data_non_object_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"successful": True, "data": [1, 2, 3]})

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="data"):
        client.get_topology()


def test_get_topology_generic_map_exception_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/ob/clusters":
            return _envelope(path, _CLUSTERS_RAW)
        if path == "/api/v2/ob/tenants":
            # 非 dict 条目 → map_topology 内 .get 抛 AttributeError（非 OcpClientError）→ 被包
            return _envelope(path, [123])
        return httpx.Response(404)

    client = RealOcpClient(_cfg(), transport=httpx.MockTransport(handler))
    with pytest.raises(OcpClientError, match="映射失败"):
        client.get_topology()


def test_base_url_blank_whitespace_raises():
    client = RealOcpClient(OcpConfig(provider="real", base_url="   "))
    with pytest.raises(OcpClientError, match="base_url"):
        client.get_topology()


def test_base_url_requires_scheme():
    client = RealOcpClient(
        _cfg(base_url="ocp.example.com"),
        transport=httpx.MockTransport(lambda req: httpx.Response(404)),
    )
    with pytest.raises(OcpClientError, match="http"):
        client.get_topology()
