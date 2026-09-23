"""OceanBase Oracle 模式租户的只读 SQL 执行器。

与 real.py 的分工：real.py 走 PyMySQL 连 MySQL 模式租户，本模块走 OCI 驱动连
Oracle 模式租户。两者对上层暴露同一个 SqlExecutor 契约，agent 不感知方言。

驱动可配置（sql_ro.driver）：
- ``oracledb``（默认）：python-oracledb，cx_Oracle 的官方后继版本。**瘦模式不需要
  任何 Oracle 客户端库**，且本机 Python 3.13 有 manylinux 轮子。
- ``cx_oracle``：需 Python <= 3.10 并安装 Oracle Instant Client。cx_Oracle 8.3.0
  （2021 年最后一版）只发布到 cp310 轮子，在 Python 3.11/3.13 上既无轮子也无
  Python.h 编译源码，因此仅在老环境里才可选。

两个驱动的 API 同源（CLOB / LONG_STRING / LONG_BINARY / cursor.var /
outputtypehandler / ping / call_timeout 均一致），所以只有装载那一步需要分支。
导入保持**惰性**（同 real.py 对 pymysql 的写法）：没装驱动只影响 Oracle 租户。

连接按执行器实例复用（长连接 + ping 自愈 + 锁串行化），理由同 real.py：一轮对话
多次调工具，每次重新 TCP + 鉴权经 obproxy 时代价明显。
"""
from __future__ import annotations

import threading
from dataclasses import replace

from app.config import SqlConfig
from app.tools.base import QueryResult, SqlExecutionError
from app.tools.sql.guard import assert_read_only, assert_safe_identifier
from app.tools.sql.registry import register

_DRIVER_HINT = (
    "缺少依赖 oracledb，请先 pip install oracledb"
    "（cx_Oracle 无 Python 3.11+ 轮子，选它需要 Python <= 3.10 + Oracle Instant Client）"
)


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


def resolve_config(cfg: SqlConfig, tenant_name: str) -> SqlConfig:
    """把 sql_ro 的通用配置解析成「本租户可直连」的 Oracle 配置。

    Oracle 模式的租户信息不写在用户名里（与 MySQL 的 ``user@tenant#cluster`` 不同），
    而是通过 DSN 的 service_name 让 ODP 路由到对应租户，用户名只带租户名。

    - username：配置里已含 ``@`` 就原样用，否则补成 ``user@tenant``
    - service_name：未显式配置时用租户名；需要 ``租户#集群`` 或自定义 service name
      时在 sql_ro.service_name 里显式指定
    """
    base = (cfg.username or "").strip()
    username = base if "@" in base else f"{base}@{tenant_name}"
    return replace(
        cfg,
        username=username,
        service_name=(cfg.service_name or "").strip() or tenant_name,
    )


def build_dsn(cfg: SqlConfig) -> str:
    """拼 DSN：``host:port/service_name``。

    OceanBase 支持直接以 ``ODP 地址:端口/SERVICE_NAME`` 登录，ODP 再据此把连接路由到
    对应租户（一个租户至多一个 SERVICE_NAME）。
    """
    return f"{cfg.host}:{cfg.port}/{cfg.service_name}"


class OBOracleSqlExecutor:
    def __init__(self, config: SqlConfig) -> None:
        # 传入的 config 应已经过 resolve_config：host/port/username/service_name 齐全
        self._cfg = config
        self._conn = None
        self._drv = None
        self._lock = threading.Lock()
        register(self)

    def _driver(self):
        if self._drv is None:
            self._drv = load_driver(self._cfg.driver)
        return self._drv

    def _connect(self):
        drv = self._driver()
        # tcp_connect_timeout：建连阶段的超时（与 pymysql 的 connect_timeout 对应）
        conn = drv.connect(
            user=self._cfg.username,
            password=self._cfg.password,
            dsn=build_dsn(self._cfg),
            tcp_connect_timeout=self._cfg.connect_timeout,
        )
        # 类型转换必须在执行任何查询之前挂上
        conn.outputtypehandler = make_type_handler(drv)
        # call_timeout 单位是毫秒（pymysql 的 read_timeout 是秒）
        conn.call_timeout = int(self._cfg.query_timeout_seconds) * 1000
        return conn

    # ---- 连接生命周期（以下方法要求调用方已持有 self._lock）----

    def _close_locked(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响后续重连
                pass

    def _conn_for_use(self):
        """复用长连接；服务端断连/空闲超时用 ping() 自愈。

        注意：cx_Oracle 的 ping() 不接受 reconnect 参数（pymysql 才有），
        所以这里靠「ping 失败就丢弃重建」达到同样效果。
        """
        if self._conn is None:
            self._conn = self._connect()
            return self._conn
        try:
            self._conn.ping()
        except Exception:  # noqa: BLE001 - 连不回来就丢弃，下次重建
            self._close_locked()
            self._conn = self._connect()
        return self._conn

    def _is_connection_error(self, exc: BaseException) -> bool:
        """只有连接类错误才丢弃连接；SQL 语法/权限等错误保留连接。"""
        drv = self._driver()
        return isinstance(exc, (drv.OperationalError, drv.InterfaceError))

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _run(self, sql: str) -> QueryResult:
        with self._lock:
            # 建连放在 try 之外：驱动缺失等错误应原样抛出，不该被当成连接类错误去重连
            conn = self._conn_for_use()
            cur = conn.cursor()
            try:
                cur.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                max_rows = self._cfg.max_rows
                # 与 real.py 一致：只多取一行判断是否截断，避免把整表拉进内存
                # （参考实现用 fetchall() 再切片，大结果集会直接吃满内存）。
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
                return QueryResult(
                    columns=cols,
                    rows=[list(r) for r in rows],
                    truncated=truncated,
                )
            except Exception as e:
                if self._is_connection_error(e):
                    self._close_locked()
                raise
            finally:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001 - 游标关闭失败不影响结果
                    pass

    def query(self, sql: str) -> QueryResult:
        assert_read_only(sql)  # 防线 1：先于连接
        try:
            return self._run(sql)
        except SqlExecutionError:
            raise
        except Exception as e:  # 连接/执行失败 → 清晰错误
            raise SqlExecutionError(f"SQL 执行失败: {e}") from e

    def explain(self, sql: str) -> QueryResult:
        """v1 与 query 同路径（与 real.py 的语义一致）。

        刻意**不**把 SQL 改写成 ``EXPLAIN PLAN FOR``：Oracle 该语句会往 PLAN_TABLE
        写入，与只读防线冲突。执行计划请走 OCP 的 get_sql_explain 工具。
        """
        assert_read_only(sql)
        try:
            return self._run(sql)
        except SqlExecutionError:
            raise
        except Exception as e:
            raise SqlExecutionError(f"EXPLAIN 失败: {e}") from e

    def table_ddl(self, table_name: str) -> QueryResult:
        """表结构：优先 DBMS_METADATA.GET_DDL（等价于 MySQL 的 SHOW CREATE TABLE）。

        部分 OceanBase Oracle 租户未开放 DBMS_METADATA，此时回退到数据字典
        （USER_TAB_COLUMNS），至少把列定义返回给模型，而不是整体报错。
        """
        safe = assert_safe_identifier(table_name, "表名")
        owner, _, obj = safe.rpartition(".")
        if not obj:  # 没有 schema 前缀：rpartition 会把整串放在 owner 里
            owner, obj = "", owner

        if owner:
            ddl_sql = f"select dbms_metadata.get_ddl('TABLE', '{obj}', '{owner}') as ddl from dual"
        else:
            ddl_sql = f"select dbms_metadata.get_ddl('TABLE', '{obj.upper()}') as ddl from dual"
        try:
            return self._run(ddl_sql)
        except SqlExecutionError:
            raise  # 驱动缺失等环境问题：回退到数据字典也救不了
        except Exception as ddl_err:
            # DBMS_METADATA 不可用（未开放/权限不足）：退回数据字典查列定义
            # Oracle 里未加引号的表名默认大写存储
            fallback_sql = (
                "select column_name, data_type, data_length, data_precision, data_scale, nullable "
                f"from user_tab_columns where table_name = '{obj.upper()}' order by column_id"
            )
            try:
                return self._run(fallback_sql)
            except Exception as e:
                raise SqlExecutionError(
                    f"获取表结构失败：DBMS_METADATA 与数据字典均不可用（{ddl_err}；{e}）"
                ) from e