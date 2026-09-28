"""OceanBase Oracle 模式租户的只读 SQL 执行器：OCI 驱动（oracledb / cx_Oracle）。

与 mysql.py（MySQL/PyMySQL）的分工：mysql.py 走 PyMySQL 连 MySQL 模式租户，本模块走
OCI 驱动连 Oracle 模式租户。两者共用 ``app.tools.sql.base.PooledSqlExecutor`` 的
长连接复用、锁串行化、只读防线与结果截断，只有「建连」与「表结构 SQL」是方言相关；
上层只看到同一个 ``SqlExecutor`` 契约，agent 不感知方言。

驱动可配置（sql_ro.driver）：
- ``oracledb``（默认）：python-oracledb，cx_Oracle 的官方后继版本。**瘦模式不需要
  任何 Oracle 客户端库**，且本机 Python 3.13 有 manylinux 轮子。
- ``cx_oracle``：需 Python <= 3.10 并安装 Oracle Instant Client。cx_Oracle 8.3.0
  （2021 年最后一版）只发布到 cp310 轮子，在 Python 3.11/3.13 上既无轮子也无
  Python.h 编译源码，因此仅在老环境里才可选。

两个驱动的 API 大体同源（CLOB / LONG_STRING / LONG_BINARY / cursor.var / outputtypehandler /
ping 均一致），差异集中在**超时**：python-oracledb 独有 ``tcp_connect_timeout``（建连），
``call_timeout``（语句，毫秒）两个驱动都有但老版本 cx_Oracle 未必可写，因此超时按驱动能力
下传、设不上只记日志（见 ``_connect``）；驱动装载那一步同样需要分支。
导入保持**惰性**（同 mysql.py 对 pymysql 的写法）：没装驱动只影响 Oracle 租户。

连接按执行器实例复用（长连接 + ping 自愈 + 锁串行化），理由同 mysql.py：一轮对话
多次调工具，每次重新 TCP + 鉴权经 obproxy 时代价明显。
"""
from __future__ import annotations

import logging
from dataclasses import replace

from app.config import SqlConfig
from app.tools.base import QueryResult, SqlExecutionError
from app.tools.sql.base import PooledSqlExecutor
from app.tools.sql.guard import assert_safe_identifier

logger = logging.getLogger(__name__)

_DRIVER_HINT = (
    "缺少依赖 oracledb，请先 pip install oracledb"
    "（cx_Oracle 无 Python 3.11+ 轮子，选它需要 Python <= 3.10 + Oracle Instant Client）"
)


def _supports_tcp_connect_timeout(drv) -> bool:
    """``tcp_connect_timeout`` 是 python-oracledb 专有的建连参数：cx_Oracle 传了会报错。

    python-oracledb 独有 ``is_thin_mode``，据此探测（cx_Oracle 侧只能靠 OS/TCP 默认超时）。
    """
    return callable(getattr(drv, "is_thin_mode", None)) or getattr(drv, "__name__", "") == "oracledb"


def load_driver(name: str):
    """按 sql_ro.driver 惰性装载 OCI 驱动模块。

    单独抽出来是为了：1) 缺驱动时给出可操作的报错；2) 测试可以注入假驱动。
    """
    key = (name or "oracledb").strip().lower()
    if key in {"oracledb", "python-oracledb"}:
        try:
            import oracledb
        except ImportError as e:  # pragma: no cover - 取决于部署环境
            raise SqlExecutionError(_DRIVER_HINT) from e
        return oracledb
    if key in {"cx_oracle", "cx-oracle", "cxoracle"}:
        try:
            import cx_Oracle
        except ImportError as e:  # pragma: no cover - 取决于部署环境
            raise SqlExecutionError(
                "缺少依赖 cx_Oracle，请安装（另需 Python <= 3.10 与 Oracle Instant Client）"
            ) from e
        return cx_Oracle
    raise SqlExecutionError(f"未知的 Oracle 驱动: {name!r}（可选 oracledb | cx_oracle）")


def make_type_handler(drv):
    """构造 outputtypehandler：把 CLOB/BLOB 拉成 Python 原生类型。

    必须有返回值：返回 cursor.var 表示「按我指定的类型取」，返回 None 表示交回
    驱动默认处理。OceanBase Oracle 模式下查表结构（DBMS_METADATA.GET_DDL）拿到的是
    CLOB —— 不转换就会得到 LOB 对象，既没法直接进 JSON，也没法给 LLM 看。
    """

    def handler(cursor, name, default_type, size, precision, scale):
        if default_type == drv.CLOB:
            return cursor.var(drv.LONG_STRING, arraysize=cursor.arraysize)
        if default_type == drv.BLOB:
            return cursor.var(drv.LONG_BINARY, arraysize=cursor.arraysize)
        return None

    return handler


def resolve_config(cfg: SqlConfig, tenant_name: str, cluster_name: str) -> SqlConfig:
    """把 sql_ro 的通用配置解析成「本租户可直连」的 Oracle 配置。

    - username：配置里已含 ``@`` 就原样用，否则补成 ``user@tenant#cluster_name``
    - service name：直接取工具传入的 ``cfg.db_name`` —— ``execute_sql`` /
      ``get_table_ddl`` 的 ``db_name`` 参数就是本租户的 service name，配置里不再有
      service name 这一项。Oracle 模式没有「库名」概念，所以解析结果写回
      ``db_name`` 供 ``build_dsn`` 使用；``db_name`` 为空时回退租户名（防御性兜底，
      正常调用不会为空）。
    """
    base = (cfg.username or "").strip()
    username = base if "@" in base else f"{base}@{tenant_name}#{cluster_name}"
    return replace(
        cfg,
        username=username,
        db_name=(cfg.db_name or "").strip() or tenant_name,
    )


def build_dsn(cfg: SqlConfig) -> str:
    """拼 DSN：``host:port/service_name``。

    OceanBase 支持直接以 ``ODP 地址:端口/SERVICE_NAME`` 登录，ODP 再据此把连接路由到
    对应租户（一个租户至多一个 SERVICE_NAME）。service name 由 ``resolve_config`` 从
    工具传入的 ``db_name`` 落到 ``cfg.db_name``，故此处读 ``db_name``。
    """
    return f"{cfg.host}:{cfg.port}/{cfg.db_name}"


class OracleSqlExecutor(PooledSqlExecutor):
    """OCI 执行器；连接生命周期与只读语义全部继承自 ``PooledSqlExecutor``。

    传入的 config 应已经过 ``resolve_config``：host/port/username 齐全，
    ``db_name`` 已落成 DSN 的 service name。
    """

    def _load_driver(self):
        return load_driver(self._cfg.driver)

    def _connect(self):
        drv = self._driver()
        kwargs = {
            "user": self._cfg.username,
            "password": self._cfg.password,
            "dsn": build_dsn(self._cfg),
        }
        if _supports_tcp_connect_timeout(drv):
            # 建连阶段的超时（与 pymysql 的 connect_timeout 对应）；cx_Oracle 无此参数
            kwargs["tcp_connect_timeout"] = self._cfg.connect_timeout
        conn = drv.connect(**kwargs)
        # 类型转换必须在执行任何查询之前挂上
        conn.outputtypehandler = make_type_handler(drv)
        # call_timeout 单位是毫秒（pymysql 的 read_timeout 是秒）。老版本 cx_Oracle 没有该
        # 可写属性，因此是尽力而为：设不上只记日志，不影响连接可用性。
        try:
            conn.call_timeout = int(self._cfg.query_timeout_seconds) * 1000
        except Exception as e:  # noqa: BLE001 - 超时降级，不能让建连失败
            logger.debug("驱动 %s 不支持 call_timeout，跳过语句超时：%s", self._cfg.driver, e)
        return conn

    def _is_connection_error(self, exc: BaseException) -> bool:
        """只有连接类错误才丢弃连接；SQL 语法/权限等错误保留连接。"""
        drv = self._driver()
        return isinstance(exc, (drv.OperationalError, drv.InterfaceError))

    def table_ddl(self, table_name: str) -> QueryResult:
        """表结构：优先 DBMS_METADATA.GET_DDL（等价于 MySQL 的 SHOW CREATE TABLE）。

        owner 恒取本连接的 ``db_name``（Oracle 模式下它同时是 DSN 的 service name 与该租户的
        schema），所以 ``schema.table`` 这种写法里的前缀不参与查询，只取对象名——否则整串
        被当成对象名，GET_DDL 找不到，回退查询也查不到，会静默返回空结果。

        部分 OceanBase Oracle 租户未开放 DBMS_METADATA，此时回退到数据字典
        （``all_tab_columns``，限定 owner；只读账号通常没有 DBA 权限，不能用 ``dba_tab_columns``），
        至少把列定义返回给模型，而不是整体报错。
        """
        raw_name = assert_safe_identifier(table_name, "表名")
        schema, _, bare_name = raw_name.rpartition(".")
        if schema:
            logger.debug("忽略表名里的 schema 前缀 %r：owner 固定取 db_name", schema)
        table_name = bare_name.upper()
        db_name = (self._cfg.db_name or "").upper()
        ddl_sql = f"select dbms_metadata.get_ddl('TABLE', '{table_name}', '{db_name}') as ddl from dual"
        try:
            return self._run(ddl_sql)
        except SqlExecutionError:
            raise  # 驱动缺失等环境问题：回退到数据字典也救不了
        except Exception as ddl_err:
            # DBMS_METADATA 不可用（未开放/权限不足）：退回数据字典查列定义
            # Oracle 里未加引号的表名默认大写存储，owner 同样大写
            fallback_sql = (
                "select column_name, data_type, data_length, data_precision, data_scale, nullable "
                f"from all_tab_columns where owner = '{db_name}' and table_name = '{table_name}' "
                "order by column_id"
            )
            try:
                return self._run(fallback_sql)
            except Exception as e:
                raise SqlExecutionError(
                    f"获取表结构失败：DBMS_METADATA 与数据字典均不可用（{ddl_err}；{e}）"
                ) from e