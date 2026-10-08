"""RealOcpClient HTTP 层测试（httpx.MockTransport 注入，不出网）。

背景：real OCP 客户端一直没有 HTTP 层测试——main 分支的 ``tests/test_real_ocp.py``（59 例）
随 ``map_*`` 映射层一起被删，留下的 ``tests/test_basic.py`` 只覆盖 ``_ms_to_us`` /
``_elapsed_us`` / ``_first_present`` / ``_normalize_mode`` 四个纯函数；``RealOcpClient`` 在
``tests/`` 里从未被实例化，``__init__(..., transport=)`` 这个注入点也就一直没人用
（``backend/app/tools/ocp/real.py`` 的 __init__ 注释就是为它写的）。

本文件补三件事：
1. **鉴权与 envelope 归一**：Basic Auth 有无、HTTP 状态码 / 传输失败 / 非 JSON / 成功位 /
   缺 data / data 非对象，一律归一成 ``OcpClientError``，不让 httpx 异常或裸 ``KeyError``
   漏到工具层。
2. **两个返回契约**：列表接口解包 ``data.contents`` 返回 ``list``，单对象接口返回 ``data``
   字典 —— 必须与 ``MockOcpClient`` 同构，因为工具层是按 mock 写的（见 tests/test_mock_ocp.py）。
3. **默认时间窗在调用时求值**：写成函数默认值会在模块导入时被固化，长驻进程会静默返回空结果。
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta

import httpx
import pytest

from app.config import OcpConfig
from app.tools.base import OcpClientError
from app.tools.ocp import real as real_module
from app.tools.ocp.mock import MockOcpClient
from app.tools.ocp.real import (
    _DEFAULT_SLOW_SQL_WINDOW_MINUTES,
    RealOcpClient,
    _now_iso,
    _window_start_iso,
)

_BASE_URL = "https://ocp.example.com"
_BASIC_AUTH = "Basic " + base64.b64encode(b"ocp:secret").decode()

# 显式时间窗：这些用例只关心路径/参数/退出码，用固定值避免与「默认窗」用例互相干扰
_T0 = "2026-01-01T00:00:00+08:00"
_T1 = "2026-01-01T01:00:00+08:00"


def _cfg(**kw) -> OcpConfig:
    defaults = {
        "provider": "real",
        "base_url": _BASE_URL,
        "username": "ocp",
        "password": "secret",
        "verify_ssl": True,
    }
    defaults.update(kw)
    return OcpConfig(**defaults)


def _client(handler, **kw) -> RealOcpClient:
    """把 handler 包成 MockTransport 注入——real.py 唯一的测试接缝。"""
    return RealOcpClient(_cfg(**kw), transport=httpx.MockTransport(handler))


def _envelope(data) -> httpx.Response:
    return httpx.Response(200, json={"successful": True, "data": data})


def _contents(items) -> httpx.Response:
    return _envelope({"contents": items})


def _recorder(response_factory):
    """返回 (handler, seen)；handler 记录每个请求、用工厂现造响应（Response 不可复用）。"""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return response_factory(request)

    return handler, seen


# ---------- 配置校验（不发请求） ----------


def test_missing_base_url_raises_before_any_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - 不该被调用
        raise AssertionError("base_url 缺失时不应发出 HTTP 请求")

    client = RealOcpClient(
        OcpConfig(provider="real", base_url=""), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(OcpClientError, match="base_url"):
        client.get_tenants_list()


def test_base_url_without_scheme_raises():
    client = _client(lambda request: _contents([]), base_url="ocp.example.com")
    with pytest.raises(OcpClientError, match="http://"):
        client.get_tenants_list()


# ---------- 鉴权 ----------


def test_requests_carry_basic_auth():
    handler, seen = _recorder(lambda request: _contents([]))
    _client(handler).get_tenants_list()
    assert len(seen) == 1
    assert seen[0].headers.get("Authorization") == _BASIC_AUTH
    assert seen[0].url.path == "/api/v2/ob/tenants"


def test_blank_username_sends_no_auth_header():
    handler, seen = _recorder(lambda request: _contents([]))
    _client(handler, username="").get_tenants_list()
    assert "Authorization" not in seen[0].headers


# ---------- 返回契约：列表接口 vs 单对象接口 ----------

_LIST_CASES = [
    ("get_slow_sql", (1, 1001, _T0, _T1), "/api/v2/ob/clusters/1/tenants/1001/slowSql"),
    ("get_sql_top_plan", (1, 1001, "sql-1", _T0, _T1), "/api/v2/ob/clusters/1/tenants/1001/sqls/sql-1/topPlan"),
    ("get_tenants_list", (), "/api/v2/ob/tenants"),
    ("get_clusters_list", (), "/api/v2/ob/clusters"),
    ("get_server_resource_stats", (1,), "/api/v2/ob/clusters/1/serverStats"),
]

_OBJECT_CASES = [
    ("get_sql_text", (1, 1001, "sql-1", _T0, _T1), "/api/v2/ob/clusters/1/tenants/1001/sqls/sql-1/text"),
    ("get_sql_explain", (1, 1001, "uid-1", _T0, _T1), "/api/v2/ob/clusters/1/tenants/1001/plans/uid-1/explain"),
    ("get_tenant_info", (1, 1001), "/api/v2/ob/clusters/1/tenants/1001"),
    ("get_cluster_resource_stats", (1,), "/api/v2/ob/clusters/1/stats"),
]


@pytest.mark.parametrize("name,args,path", _LIST_CASES, ids=[c[0] for c in _LIST_CASES])
def test_list_endpoints_unwrap_contents_into_a_plain_list(name, args, path):
    handler, seen = _recorder(lambda request: _contents([{"id": 1}, {"id": 2}]))
    result = getattr(_client(handler), name)(*args)
    assert result == [{"id": 1}, {"id": 2}]  # 解的是 contents，不是整个 data
    assert [r.url.path for r in seen] == [path]


@pytest.mark.parametrize("name,args,path", _OBJECT_CASES, ids=[c[0] for c in _OBJECT_CASES])
def test_object_endpoints_return_the_data_dict(name, args, path):
    handler, seen = _recorder(lambda request: _envelope({"id": 7, "name": "obj"}))
    result = getattr(_client(handler), name)(*args)
    assert result == {"id": 7, "name": "obj"}  # 单对象接口不套 contents
    assert [r.url.path for r in seen] == [path]


def test_empty_contents_is_an_empty_list_not_an_error():
    handler, _ = _recorder(lambda request: _contents([]))
    assert _client(handler).get_tenants_list() == []
    assert _client(handler).get_slow_sql(1, 1001, _T0, _T1) == []


def test_contract_table_matches_mock_client():
    """本文件的类型表必须与 MockOcpClient 一致——mock 才是工具层面对的权威契约。"""
    mock = MockOcpClient()
    assert isinstance(mock.get_slow_sql(1, "1001", _T0, _T1), list)
    assert isinstance(mock.get_sql_top_plan(1, "1001", "sql-1", _T0, _T1), list)
    assert isinstance(mock.get_tenants_list(), list)
    assert isinstance(mock.get_clusters_list(), list)
    assert isinstance(mock.get_server_resource_stats(1), list)
    assert isinstance(mock.get_sql_text(1, "1001", "sql-1", _T0, _T1), dict)
    assert isinstance(mock.get_sql_explain(1, "1001", "uid-1", _T0, _T1), dict)
    assert isinstance(mock.get_tenant_info(1, "1001"), dict)
    assert isinstance(mock.get_cluster_resource_stats(1), dict)
    # 与本文件两张表逐一对照，任一侧改了形状都会红
    assert {name for name, _, _ in _LIST_CASES} == {
        "get_slow_sql",
        "get_sql_top_plan",
        "get_tenants_list",
        "get_clusters_list",
        "get_server_resource_stats",
    }
    assert {name for name, _, _ in _OBJECT_CASES} == {
        "get_sql_text",
        "get_sql_explain",
        "get_tenant_info",
        "get_cluster_resource_stats",
    }


# ---------- envelope / 传输错误归一 ----------

_FAILURE_CASES = [
    ("http_404", lambda: httpx.Response(404, text="no such tenant"), "404"),
    ("http_500_body_in_message", lambda: httpx.Response(500, text="ob cluster not ready"), "ob cluster not ready"),
    ("unsuccessful_with_reason", lambda: httpx.Response(200, json={"successful": False, "error": "tenant not found"}), "tenant not found"),
    ("unsuccessful_without_reason", lambda: httpx.Response(200, json={"successful": False}), "unsuccessful"),
    ("successful_without_data", lambda: httpx.Response(200, json={"successful": True}), "缺 data"),
    ("data_is_a_list", lambda: httpx.Response(200, json={"successful": True, "data": []}), "非对象"),
    ("non_json_body", lambda: httpx.Response(200, content=b"<html>bad gateway</html>"), "非 JSON"),
    ("contents_missing", lambda: httpx.Response(200, json={"successful": True, "data": {"foo": 1}}), "contents"),
    ("contents_not_a_list", lambda: httpx.Response(200, json={"successful": True, "data": {"contents": {"id": 1}}}), "contents"),
]


@pytest.mark.parametrize("name,factory,match", _FAILURE_CASES, ids=[c[0] for c in _FAILURE_CASES])
def test_failures_are_normalized_to_ocp_client_error(name, factory, match):
    handler, _ = _recorder(lambda request: factory())
    with pytest.raises(OcpClientError, match=match):
        _client(handler).get_tenants_list()


def test_transport_error_is_normalized():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect boom")

    with pytest.raises(OcpClientError, match="connect boom"):
        _client(handler).get_tenants_list()


def test_ocp_client_error_is_a_value_error():
    """工具层按 ValueError 之外无差别捕获；这条锁住异常继承关系别被改掉。"""
    assert issubclass(OcpClientError, ValueError)


# ---------- 时间窗：默认值必须调用时求值 ----------

_WINDOWED_CASES = [
    ("get_slow_sql", (1, 1001)),
    ("get_sql_text", (1, 1001, "sql-1")),
    ("get_sql_explain", (1, 1001, "uid-1")),
    ("get_sql_top_plan", (1, 1001, "sql-1")),
]


@pytest.mark.parametrize("name,args", _WINDOWED_CASES, ids=[c[0] for c in _WINDOWED_CASES])
def test_default_window_is_resolved_per_call_not_at_import(monkeypatch, name, args):
    """默认窗取值必须走模块级函数（每次调用重新算）。

    旧的 ``start_time: str = datetime.now(...).isoformat()`` 写法在模块导入时就把窗口
    钉死了——monkeypatch 打桩函数对它完全无效，这条用例会红。
    """
    monkeypatch.setattr(real_module, "_window_start_iso", lambda *a, **k: "START-SENTINEL")
    monkeypatch.setattr(real_module, "_now_iso", lambda: "END-SENTINEL")
    handler, seen = _recorder(lambda request: _contents([]))
    getattr(_client(handler), name)(*args)
    params = seen[0].url.params
    assert params["startTime"] == "START-SENTINEL"
    assert params["endTime"] == "END-SENTINEL"


@pytest.mark.parametrize("name,args", _WINDOWED_CASES, ids=[c[0] for c in _WINDOWED_CASES])
def test_explicit_window_wins_over_the_default(monkeypatch, name, args):
    def explode(*a, **k):  # pragma: no cover - 显式传窗时不该再算默认值
        raise AssertionError("显式传入时间窗时不应再求默认值")

    monkeypatch.setattr(real_module, "_window_start_iso", explode)
    monkeypatch.setattr(real_module, "_now_iso", explode)
    handler, seen = _recorder(lambda request: _contents([]))
    getattr(_client(handler), name)(*args, _T0, _T1)
    params = seen[0].url.params
    assert params["startTime"] == _T0
    assert params["endTime"] == _T1


def test_default_window_covers_the_last_thirty_minutes():
    """真实默认窗：起点 = 调用时刻 - 30min，终点 = 调用时刻。"""
    start = datetime.fromisoformat(_window_start_iso())
    end = datetime.fromisoformat(_now_iso())
    assert end - start == timedelta(minutes=_DEFAULT_SLOW_SQL_WINDOW_MINUTES)
    assert abs((datetime.now(end.tzinfo) - end).total_seconds()) < 5


# ---------- 参数透传 ----------


def test_slow_sql_passes_through_the_optional_filters():
    handler, seen = _recorder(lambda request: _contents([]))
    _client(handler).get_slow_sql(
        1,
        1001,
        _T0,
        _T1,
        server_id=7,
        inner=True,
        sql_text="orders",
        filter_expression="@avgElapsedTime>=1000",
        limit=5,
        sql_text_length=500,
    )
    params = seen[0].url.params
    assert params["startTime"] == _T0
    assert params["endTime"] == _T1
    assert params["sqlTextLength"] == "500"
    assert params["limit"] == "5"
    assert params["serverId"] == "7"
    assert params["sqlText"] == "orders"
    assert params["filterExpression"] == "@avgElapsedTime>=1000"
    assert params["inner"] in ("True", "true")


def test_unset_optional_filters_are_not_sent():
    """None 的可选参数必须整条不发。

    httpx 会把 ``{k: None}`` 写成 ``k=``（空串）而不是省略，空 ``filterExpression`` /
    空 ``limit`` 在 OCP 侧容易被当成非法入参 400 —— 见 real.py::_clean_params。
    """
    handler, seen = _recorder(lambda request: _contents([]))
    _client(handler).get_slow_sql(1, 1001, _T0, _T1)
    params = seen[0].url.params
    assert params["sqlTextLength"] == "200"  # 模块默认值
    assert "filterExpression" not in params
    assert "serverId" not in params
    assert "sqlText" not in params
    assert "limit" not in params
