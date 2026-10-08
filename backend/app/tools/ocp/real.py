"""OCP real：真实 OCP 报文+ httpx HTTP 客户端。

模块分层：
- 模块级常量：默认超时
- RealOcpClient：httpx HTTP 客户端（Basic Auth + envelope 归一），
  网络失败 / 报文异常 / 映射阶段异常统一归一为 OcpClientError。

两个返回契约（与 ``app/tools/ocp/mock.py`` 严格对齐，工具层按 mock 编写）：
- **列表接口**（slowSql / topPlan / tenants / clusters / serverStats）：OCP 把数组放在
  ``data.contents``，走 ``_get_contents()``，校验后返回 ``list``。
- **单对象接口**（sqls/{id}/text、plans/{uid}/explain、单租户详情、集群 stats）：OCP 把对象
  直接放在 ``data``，走 ``_get_data()``，返回 ``dict``。

参考 OCP 4.3.5 文档：https://www.oceanbase.com/docs/common-ocp-1000000002381372

"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.config import OcpConfig, load_settings
from app.tools.base import OcpClientError
from app.tools.ocp import get_ocp_client

# ---------- 模块级常量 ----------

_DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
_DEFAULT_SLOW_SQL_WINDOW_MINUTES = 30
_DEFAULT_SLOW_SQL_TEXT_LENGTH = 200
_SYSTEM_TENANT_NAMES = {"sys", "ocpmeta", "ocpmonitor"}
_CLUSTER_ID_KEYS = ("clusterId", "obClusterId", "cluster_id")
_OCP_TZ = ZoneInfo("Asia/Shanghai")


# ---------- 纯辅助函数 ----------


def _now_iso() -> str:
    """当前时间（Asia/Shanghai）ISO 串。

    必须在**调用时**求值。写成 ``end_time: str = datetime.now(...).isoformat()`` 形式的
    函数默认值会在模块导入时算一次并被永久固化，长驻进程（uvicorn）跑久了时间窗就停在
    启动那一刻，表现为慢 SQL 接口**静默返回空结果**——同一个坑见
    ``app/agent/tool_input.py:3-5`` 的告诫。
    """
    return datetime.now(_OCP_TZ).isoformat(timespec="seconds")


def _window_start_iso(minutes: int = _DEFAULT_SLOW_SQL_WINDOW_MINUTES) -> str:
    """默认时间窗起点（调用时求值，理由同 ``_now_iso``）。"""
    return (datetime.now(_OCP_TZ) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def _first_present(raw: dict, keys: tuple[str, ...], default: str = "") -> str:
    """按序返回首个非空键值并转 str；全空（缺键/None/""）返回 default。"""
    for k in keys:
        v = raw.get(k)
        if v not in (None, ""):
            return str(v)
    return default


def _ms_to_us(ms) -> int:
    """OCP 毫秒耗时 → 微秒（模型内统一 us）。None/空 → 0。"""
    if ms is None:
        return 0
    if isinstance(ms, str) and not ms.strip():
        return 0
    return int(round(float(ms) * 1000))


def _elapsed_us(raw: dict, camel_ms_key: str, us_key: str, ms_key: str) -> int:
    """耗时字段键容忍映射，统一输出微秒。

    键优先级：camel 毫秒键存在(非 None) → _ms_to_us；否则 us 键存在 → 视为已 us 透传；
    否则 *_ms 键存在 → _ms_to_us；否则 0。
    """
    v = raw.get(camel_ms_key)
    if v is not None:
        return _ms_to_us(v)
    v = raw.get(us_key)
    if v is not None:
        if isinstance(v, str) and not v.strip():
            return 0
        return int(round(float(v)))
    v = raw.get(ms_key)
    if v is not None:
        return _ms_to_us(v)
    return 0


def _clean_params(params: dict | None) -> dict | None:
    """丢掉值为 None 的可选参数。

    httpx 对 ``{k: None}`` 的序列化结果是 ``k=``（空串）而**不是**省略该参数，于是
    ``filterExpression=&limit=&serverId=&sqlText=`` 会被发出去 —— 空表达式/空数字在 OCP
    侧容易被当成非法入参直接 400。None 的语义是「调用方没指定」，就该整条不发。
    """
    if not params:
        return params
    return {k: v for k, v in params.items() if v is not None}


def _normalize_mode(v) -> str:
    """归一化租户兼容模式。仅接受 mysql/oracle（忽略大小写与首尾空白）。

    mode 缺失或未知即 schema 漂移，宁可显式暴露而非静默兜底。
    """
    if v is None:
        raise OcpClientError(f"未知/缺失 OCP 租户 mode: {v!r}")
    s = str(v).strip().upper()
    if s == "MYSQL":
        return "mysql"
    if s == "ORACLE":
        return "oracle"
    raise OcpClientError(f"未知/缺失 OCP 租户 mode: {v!r}")


class RealOcpClient:
    """OCP real HTTP 客户端：Basic Auth + envelope 归一 。

    异常统一：配置/依赖缺失、HTTP/传输失败、报文形状异常、映射阶段异常全部归一为
    OcpClientError；成功时返回归一化后的 dict / list（字段与 OCP 报文同构，由工具层再加工）。
    httpx 在 __init__ 不 import（模块顶层不依赖 httpx，离线可 import），仅在使用时懒加载。
    """

    def __init__(self, config: OcpConfig, *, transport=None) -> None:
        # transport 为 httpx.MockTransport 注入点（生产 None），缓存以便 _client() 复用。
        self._cfg = config
        self._transport = transport
        self._client_cache = None

    def _require_base_url(self) -> str:
        base_url = (self._cfg.base_url or "").strip()
        if not base_url:
            raise OcpClientError("OCP real 模式未配置 base_url")
        if not base_url.startswith(("http://", "https://")):
            raise OcpClientError(
                f"OCP real base_url 需以 http:// 或 https:// 开头：{base_url!r}"
            )
        return base_url

    def _client(self):
        """懒构造并缓存 httpx.Client（Basic Auth / verify / timeout / 注入 transport）。"""
        if self._client_cache is None:
            try:
                import httpx
            except ImportError as e:  # pragma: no cover - 离线环境
                raise OcpClientError(
                    "缺少依赖 httpx，请先安装（uv add httpx / pip install httpx）"
                ) from e
            self._httpx = httpx  # 缓存模块引用，_fetch 用 self._httpx.* 引用异常类
            auth = (
                httpx.BasicAuth(self._cfg.username, self._cfg.password)
                if self._cfg.username
                else None
            )
            self._client_cache = httpx.Client(
                base_url=self._require_base_url(),
                auth=auth,
                verify=self._cfg.verify_ssl,
                timeout=_DEFAULT_HTTP_TIMEOUT_SECONDS,
                transport=self._transport,
            )
        return self._client_cache

    def _fetch(self, path: str, *, params: dict | None = None) -> dict:
        """唯一 HTTP 出口：GET + HTTP/传输/JSON/报文归一，成功返回 data 字典。"""
        client = self._client()
        try:
            response = client.get(path, params=_clean_params(params))
            response.raise_for_status()
        except self._httpx.HTTPStatusError as e:
            # 附响应体前 200 字符，联调期可直接看到服务端诊断
            status = e.response.status_code if e.response is not None else "unknown"
            text = e.response.text[:200] if e.response is not None else ""
            detail = f": {text}" if text else ""
            raise OcpClientError(f"OCP HTTP {status} on GET {path}{detail}") from e
        except self._httpx.TransportError as e:
            raise OcpClientError(f"OCP HTTP 请求失败 GET {path}: {e}") from e
        try:
            body = response.json()
        except ValueError as e:
            raise OcpClientError(f"OCP 返回非 JSON (GET {path})") from e
        if not isinstance(body, dict) or body.get("successful") is not True:
            # 保留服务端原因（error/message），便于联调期定位
            extra = body.get("error") or body.get("message") if isinstance(body, dict) else None
            suffix = f": {extra}" if extra else ""
            raise OcpClientError(f"OCP 返回 unsuccessful (GET {path}){suffix}")
        if "data" not in body:
            raise OcpClientError(f"OCP 报文缺 data 键 (GET {path})")
        data = body["data"]
        if not isinstance(data, dict):
            raise OcpClientError(f"OCP data 非对象 (GET {path})")
        return data

    def _get_data(self, path: str, *, params: dict | None = None) -> dict:
        """GET 单对象接口：返回 envelope 里的 data 字典原样（见模块 docstring 的契约表）。"""
        return self._fetch(path, params=params)

    def _get_contents(self, path: str, *, params: dict | None = None) -> list:
        """GET 列表接口：校验 data.contents 为 list 后返回（空列表合法）。

        校验必须落在这一层：OCP 在「确实没数据」和「报文漂移」两种情况下都可能不给
        contents，若下游客直接 ``data['contents']`` 会抛裸 KeyError，绕过本模块「一切异常
        归一为 OcpClientError」的承诺，工具层就再也拿不到可读的失败原因。
        """
        data = self._fetch(path, params=params)
        contents = data.get("contents")
        if not isinstance(contents, list):
            raise OcpClientError(f"OCP data.contents 缺失/非列表 (GET {path})")
        return contents

    def get_slow_sql(self, cluster_id: int, tenant_id: int,
                     start_time: str = "", end_time: str = "",
                     server_id: int = None, inner: bool = False, sql_text: str = None,
                     filter_expression: str = None, limit: int = None,
                     sql_text_length: int = _DEFAULT_SLOW_SQL_TEXT_LENGTH) -> list:
        """
        获取单租户慢sql列表
        :param sql_text:SQL 包含的关键词，关键词不区分大小写。
        :param filter_expression:所有字段通过 @ 来引用，可选字段请参考 查询 SQL 的性能统计 接口返回的所有列。
        :param limit:返回的 TOP 数目
        :param sql_text_length:返回 SQL 文本的最大长度。
        :param inner:是否为内部 SQL。
        :param server_id:查询在指定 OceanBase 服务器上的计划的性能。不指定时，查询 SQL 在所有服务器上的计划的性能。
        :param cluster_id:集群的 ID。
        :param tenant_id:租户的 ID。
        :param start_time:起始时间，形如 2026-02-16T05:32:16+08:00；留空取调用时刻往前 30 分钟。
        :param end_time:结束时间，形如 2026-02-16T05:32:16+08:00；留空取调用时刻。
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/slowSql"
        return self._get_contents(
            path,
            params={
                "startTime": start_time or _window_start_iso(),
                "endTime": end_time or _now_iso(),
                "sqlTextLength": sql_text_length,
                "filterExpression": filter_expression,
                "limit": limit,
                "inner": inner,
                "serverId": server_id,
                "sqlText": sql_text
            },
        )

    def get_sql_text(self, cluster_id: int, tenant_id: int, sql_id: str,
                     start_time: str = "", end_time: str = "") -> dict:
        """
        获取完整sql文本
        :param cluster_id:
        :param tenant_id:
        :param sql_id:
        :param start_time:留空取调用时刻往前 30 分钟。
        :param end_time:留空取调用时刻。
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/sqls/{sql_id}/text"
        return self._get_data(
            path,
            params={
                "startTime": start_time or _window_start_iso(),
                "endTime": end_time or _now_iso()
            },
        )

    def get_sql_explain(self, cluster_id: int, tenant_id: int, uid: str,
                        start_time: str = "", end_time: str = "") -> dict:
        """
        获取某段时间内指定uid的执行计划算子
        :param start_time:留空取调用时刻往前 30 分钟。
        :param end_time:留空取调用时刻。
        :param cluster_id: 集群的 ID
        :param tenant_id: 租户的 ID
        :param uid: 计划id
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/plans/{uid}/explain"
        return self._get_data(
            path,
            params={
                "startTime": start_time or _window_start_iso(),
                "endTime": end_time or _now_iso()
            },
        )

    def get_sql_top_plan(self, cluster_id: int, tenant_id: int, sql_id: str,
                         start_time: str = "", end_time: str = "") -> list:
        """
        获取某段时间内指定sql_id所产生执行计划的uid。uid用户获取执行计划算子
        :param cluster_id:
        :param tenant_id:
        :param sql_id:
        :param start_time:留空取调用时刻往前 30 分钟。
        :param end_time:留空取调用时刻。
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/sqls/{sql_id}/topPlan"
        return self._get_contents(
            path,
            params={
                "startTime": start_time or _window_start_iso(),
                "endTime": end_time or _now_iso()
            },
        )

    def get_tenants_list(self) -> list:
        """
        查询全部租户列表
        :return:
        """
        self._require_base_url()
        return self._get_contents("/api/v2/ob/tenants")

    def get_tenant_info(self, cluster_id, tenant_id) -> dict:
        """
        查询指定租户详情
        :param cluster_id:
        :param tenant_id:
        :return:
        """
        self._require_base_url()
        return self._get_data(f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}")

    def get_clusters_list(self) -> list:
        """
        查询 OCP 所管理的 OceanBase 集群信息
        :return:
        """
        self._require_base_url()
        return self._get_contents("/api/v2/ob/clusters")

    def get_cluster_resource_stats(self, cluster_id) -> dict:
        """
        获取 OceanBase 集群资源统计信息
        :param cluster_id:
        :return:
        """
        self._require_base_url()
        return self._get_data(f"/api/v2/ob/clusters/{cluster_id}/stats")

    def get_server_resource_stats(self, cluster_id) -> list:
        """
        获取集群内所有 OBServer 资源统计信息
        :param cluster_id:
        :return:
        """
        self._require_base_url()
        return self._get_contents(f"/api/v2/ob/clusters/{cluster_id}/serverStats")


if __name__ == '__main__':
    settings = load_settings()
    ocp = get_ocp_client(settings)
    data = ocp.get_slow_sql(1000001, 1000039)
    print(data)
