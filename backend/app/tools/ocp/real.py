"""OCP real：真实 OCP 报文+ httpx HTTP 客户端。

模块分层：
- 模块级常量：默认超时
- RealOcpClient：httpx HTTP 客户端（Basic Auth + envelope 归一），
  网络失败 / 报文异常 / 映射阶段异常统一归一为 OcpClientError。

参考 OCP 4.3.5 文档：https://www.oceanbase.com/docs/common-ocp-1000000002381372

"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.config import OcpConfig, load_settings
from app.tools.base import (
    ClusterInfo,
    OcpClientError
)
from app.tools.ocp import get_ocp_client

# ---------- 模块级常量 ----------

_DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
_DEFAULT_SLOW_SQL_WINDOW_MINUTES = 30
_DEFAULT_SLOW_SQL_TEXT_LENGTH = 200
_SYSTEM_TENANT_NAMES = {"sys", "ocpmeta", "ocpmonitor"}
_CLUSTER_ID_KEYS = ("clusterId", "obClusterId", "cluster_id")


# ---------- 纯辅助函数 ----------


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


def _iso_utc(dt: datetime) -> str:
    """datetime → 'YYYY-MM-DDTHH:MM:SSZ'（UTC，无小数秒，结尾 Z）。"""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RealOcpClient:
    """OCP real HTTP 客户端：Basic Auth + envelope 归一 。

    异常统一：配置/依赖缺失、HTTP/传输失败、报文形状异常、映射阶段异常全部归一为
    OcpClientError；成功返回共享模型（TopologyInfo / SlowSqlItem）。
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
            response = client.get(path, params=params)
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

    def _get_contents(self, path: str, *, params: dict | None = None) -> dict:
        """GET 列表接口：校验 data.contents 为 list 后返回（空列表合法）。"""
        row_data = self._fetch(path, params=params)
        # contents = data.get("contents")
        # if contents is None or not isinstance(contents, list):
        #     raise OcpClientError(f"OCP data.contents 缺失/非列表 (GET {path})")
        return row_data



    def get_slow_sql(self,cluster_id: int, tenant_id: int,
                     start_time: str = (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=_DEFAULT_SLOW_SQL_WINDOW_MINUTES)).isoformat(
                         timespec='seconds'),
                     end_time: str = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds'),
                     server_id: int = None, inner: bool = False, sql_text: str = None,
                     filter_expression: str = None, limit: int = None, sql_text_length: int = _DEFAULT_SLOW_SQL_TEXT_LENGTH):
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
        :param start_time:查看慢 SQL 历史参数的起始时间。该时间只支持 UTC 时间，格式为：YYYY-MM-DDThh:mm:ssZ。
        :param end_time:查看慢 SQL 历史参数的结束时间。该时间只支持 UTC 时间，格式为：YYYY-MM-DDThh:mm:ssZ。
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/slowSql"
        raw_data = self._get_contents(
            path,
            params={
                "startTime": start_time,
                "endTime": end_time,
                "sqlTextLength": sql_text_length,
                "filterExpression": filter_expression,
                "limit": limit,
                "inner": inner,
                "serverId": server_id,
                "sqlText": sql_text
            },
        )
        return raw_data['contents']

    def get_sql_text(self,cluster_id: int, tenant_id: int, sql_id: str,
                     start_time: str = (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=_DEFAULT_SLOW_SQL_WINDOW_MINUTES)).isoformat(
                         timespec='seconds'),
                     end_time: str = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')):
        """
        获取完整sql文本
        :param cluster_id:
        :param tenant_id:
        :param sql_id:
        :param start_time:
        :param end_time:
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/sqls/{sql_id}/text"
        raw_data = self._get_contents(
            path,
            params={
                "startTime": start_time,
                "endTime": end_time
            },
        )
        return raw_data

    def get_sql_explain(self, cluster_id: int, tenant_id: int, uid: str,
                        start_time: str = (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=_DEFAULT_SLOW_SQL_WINDOW_MINUTES)).isoformat(
                            timespec='seconds'),
                        end_time: str = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')):
        """
        获取某段时间内指定uid的执行计划算子
        :param start_time:
        :param end_time:
        :param cluster_id: 集群的 ID
        :param tenant_id: 租户的 ID
        :param uid: 计划id
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/plans/{uid}/explain"
        raw_data = self._get_contents(
            path,
            params={
                "startTime": start_time,
                "endTime": end_time
            },
        )
        return raw_data

    def get_sql_top_plan(self,cluster_id: int, tenant_id: int, sql_id: str,
                         start_time: str = (datetime.now(ZoneInfo("Asia/Shanghai")) - timedelta(minutes=30)).isoformat(
                             timespec='seconds'),
                         end_time: str = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec='seconds')):
        """
        获取某段时间内指定sql_id所产生执行计划的uid。uid用户获取执行计划算子
        :param cluster_id:
        :param tenant_id:
        :param sql_id:
        :param start_time:
        :param end_time:
        :return:
        """
        self._require_base_url()
        path = f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}/sqls/{sql_id}/topPlan"
        raw_data = self._get_contents(
            path,
            params={
                "startTime": start_time,
                "endTime": end_time
            },
        )

        return raw_data['contents']

    def get_tenants_list(self):
        """
        查询全部租户列表
        :return:
        """
        self._require_base_url()
        raw_data = self._get_contents(f"/api/v2/ob/tenants")
        return raw_data['contents']

    def get_tenant_info(self,cluster_id, tenant_id):
        """
        查询指定租户详情
        :param cluster_id:
        :param tenant_id:
        :return:
        """
        self._require_base_url()
        raw_data = self._get_contents(f"/api/v2/ob/clusters/{cluster_id}/tenants/{tenant_id}")
        return raw_data

    def get_clusters_list(self):
        """
        查询 OCP 所管理的 OceanBase 集群信息
        :return:
        """
        self._require_base_url()
        raw_data = self._get_contents(f"/api/v2/ob/clusters")
        return raw_data['contents']

    def get_cluster_resource_stats(self,cluster_id):
        """
        获取 OceanBase 集群资源统计信息
        :param cluster_id:
        :return:
        """
        self._require_base_url()
        raw_data = self._get_contents(f"/api/v2/ob/clusters/{cluster_id}/stats")
        return raw_data

    def get_server_resource_stats(self,cluster_id):
        """
        获取集群内所有 OBServer 资源统计信息
        :param cluster_id:
        :return:
        """
        self._require_base_url()
        raw_data = self._get_contents(f"/api/v2/ob/clusters/{cluster_id}/serverStats")
        return raw_data['contents']



if __name__ == '__main__':
    settings = load_settings()
    ocp = get_ocp_client(settings)
    data = ocp.get_slow_sql(1000001,1000039)
    print(data)