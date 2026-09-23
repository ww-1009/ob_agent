"""真实 SQL 执行器的公共骨架（方言无关）。

``MysqlSqlExecutor``（PyMySQL，mysql.py）与 ``OracleSqlExecutor``（OCI，oracle.py）
除了「怎么建连」和「表结构 SQL 怎么写」之外，其余行为完全一致：

- 只读三层防线的第 1 层：``guard.assert_read_only`` 在任何连接之前执行；
- 会话与执行防护：autocommit / 超时 / ``max_rows`` 截断（多取一行判断是否截断）；
- 长连接按执行器实例复用，``ping`` 自愈，只有断连类错误才丢弃连接；
- 每个执行器一把锁，把自己的查询串行化（驱动连接不是线程安全的）；
- 统一的错误包装：连接/执行失败一律转成 ``SqlExecutionError``。

把这些骨架收在本模块，子类只实现方言钩子，避免 mysql.py 与 oracle.py 逐行重复。
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from app.config import SqlConfig
from app.tools.base import QueryResult, SqlExecutionError
from app.tools.sql.guard import assert_read_only
from app.tools.sql.registry import register


class PooledSqlExecutor(ABC):
    """长连接复用 + 锁串行化 + 只读/截断统一语义的真实执行器基类。

    子类必须实现的方言钩子：

    - ``_load_driver()``：惰性装载驱动模块，缺依赖时抛 ``SqlExecutionError``；
    - ``_connect()``：用 ``self._cfg`` 建立一条物理连接；
    - ``_is_connection_error(exc)``：判断异常是否属于断连类；
    - ``table_ddl(table_name)``：方言相关的表结构查询（``SqlExecutor`` 契约之一）。

    可选覆盖：``_ping(conn)``（默认 ``conn.ping()``，不带 reconnect 参数）。
    """

    def __init__(self, config: SqlConfig) -> None:
        self._cfg = config
        self._conn = None
        self._drv = None  # 驱动模块缓存：整个实例只 import 一次
        self._lock = threading.Lock()
        register(self)

    # ---- 方言钩子（子类实现）------------------------------------------------

    @abstractmethod
    def _load_driver(self):
        """惰性装载驱动模块；缺依赖时抛 ``SqlExecutionError``。"""

    @abstractmethod
    def _connect(self):
        """用 ``self._cfg`` 建立一条物理连接。"""

    @abstractmethod
    def _is_connection_error(self, exc: BaseException) -> bool:
        """异常是否属于断连类：是则丢弃连接下次重建，否则保留连接。"""

    @abstractmethod
    def table_ddl(self, table_name: str) -> QueryResult:
        """表结构查询。方言不同（SHOW CREATE TABLE vs DBMS_METADATA.GET_DDL）。"""

    def _ping(self, conn) -> None:
        """默认 ``ping()`` 不接受 reconnect 参数（Oracle/cx_Oracle 即如此）。

        MySQL 的 PyMySQL 支持原地重连，子类可覆盖为 ``ping(reconnect=True)``。
        """
        conn.ping()

    def _driver(self):
        """返回缓存的驱动模块，首次访问时惰性装载。"""
        if self._drv is None:
            self._drv = self._load_driver()
        return self._drv

    # ---- 连接生命周期（以下方法要求调用方已持有 self._lock）------------------

    def _close_locked(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响后续重连
                pass

    def _conn_for_use(self):
        """复用长连接；服务端断连/空闲超时用 ping 自愈，连不回来就丢弃重建。"""
        if self._conn is None:
            self._conn = self._connect()
            return self._conn
        try:
            self._ping(self._conn)
        except Exception:  # noqa: BLE001 - 连不回来就丢弃，下次重建
            self._close_locked()
            self._conn = self._connect()
        return self._conn

    def close(self) -> None:
        """关闭长连接（应用关闭时由 registry 统一调用）。"""
        with self._lock:
            self._close_locked()

    # ---- SqlExecutor 契约 ---------------------------------------------------

    def _run(self, sql: str) -> QueryResult:
        with self._lock:
            # 建连放在 try 之外：驱动缺失等错误应原样抛出，不该被当成连接类错误去重连
            conn = self._conn_for_use()
            cur = conn.cursor()
            try:
                cur.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                max_rows = self._cfg.max_rows
                # 与原有实现一致：只多取一行判断是否截断，避免把整表拉进内存
                # （fetchall() 再切片在大结果集会直接吃满内存）。
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
                return QueryResult(columns=cols, rows=[list(r) for r in rows], truncated=truncated)
            except Exception as e:
                # 断连/超时等连接类错误：丢弃连接以便下次重连；SQL 本身的问题保留连接
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
        """v1 与 query 同路径：返回查询行而非执行计划（与 mock 的 explain 语义不同）。

        刻意**不**改写成 ``EXPLAIN PLAN FOR``：Oracle 该语句会往 PLAN_TABLE 写入，
        与只读防线冲突。执行计划请走 OCP 的 get_sql_explain 工具。
        """
        assert_read_only(sql)
        try:
            return self._run(sql)
        except SqlExecutionError:
            raise
        except Exception as e:
            raise SqlExecutionError(f"EXPLAIN 失败: {e}") from e