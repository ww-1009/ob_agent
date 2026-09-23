"""Real SQL 执行器：PyMySQL 直连，带只读三层防线。

三层防线：
1. 语法白名单（guard.assert_read_only），在任何连接之前执行。
2. 只读专用账号：建议 sql_ro 配置仅授 SELECT 的账号。
3. 会话与执行防护：autocommit=1；超时；结果行数上限截断。

连接按执行器实例**复用**（长连接 + ping(reconnect=True)）：早先每次查询都新建并关闭连接，
一轮对话多次调用工具就要多次 TCP + 鉴权（经 obproxy 时延迟明显）。PyMySQL 连接不是线程
安全的，因此每个执行器持有一把锁，把自己的查询串行化。
"""
from __future__ import annotations

import threading

from app.config import SqlConfig
from app.tools.base import QueryResult, SqlExecutionError
from app.tools.sql.guard import assert_read_only, assert_safe_identifier

# 长连接注册表（MySQL 与 Oracle 执行器共用）。close_all_executors 在此一并重新
# 导出，保持 `from app.tools.sql.real import close_all_executors` 这类旧调用方可用；
# 应用装配请直接从 app.tools.sql.registry 导入。
from app.tools.sql.registry import close_all_executors, register

__all__ = ["RealSqlExecutor", "close_all_executors"]


class RealSqlExecutor:
    def __init__(self, config: SqlConfig) -> None:
        self._cfg = config
        self._conn = None
        self._lock = threading.Lock()
        register(self)

    def _connect(self):
        try:
            import pymysql
        except ImportError as e:  # pragma: no cover
            raise SqlExecutionError("缺少依赖 pymysql，请安装") from e

        # TODO(联调): 大结果集可切 SSCursor 服务端游标逐批拉取（现为客户端 fetchmany+截断）；
        # 服务端超时 hint（ob_query_timeout）与客户端 read_timeout 的配合待联调确认。
        return pymysql.connect(
            host=self._cfg.host,
            port=self._cfg.port,
            user=self._cfg.username,
            password=self._cfg.password,
            database=self._cfg.db_name,
            connect_timeout=self._cfg.connect_timeout,
            autocommit=True,
            read_timeout=self._cfg.query_timeout_seconds,
            write_timeout=self._cfg.query_timeout_seconds,
        )

    # ---- 连接生命周期（以下方法要求调用方已持有 self._lock）----

    def _close_locked(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响后续重连
                pass

    def _conn_for_use(self):
        """复用长连接；服务端断连/空闲超时用 ping(reconnect=True) 自愈。"""
        if self._conn is None:
            self._conn = self._connect()
            return self._conn
        try:
            self._conn.ping(reconnect=True)
        except Exception:  # noqa: BLE001 - 连不回来就丢弃，下次重建
            self._close_locked()
            self._conn = self._connect()
        return self._conn

    @staticmethod
    def _is_connection_error(exc: BaseException) -> bool:
        """只有连接类错误才丢弃连接；SQL 语法/权限等错误保留连接。"""
        try:
            import pymysql
        except ImportError:  # pragma: no cover
            return False
        return isinstance(exc, (pymysql.err.OperationalError, pymysql.err.InterfaceError))

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _run(self, sql: str) -> QueryResult:
        with self._lock:
            conn = self._conn_for_use()
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cols = [d[0] for d in cur.description] if cur.description else []
                    max_rows = self._cfg.max_rows
                    rows = cur.fetchmany(max_rows + 1)
                    truncated = len(rows) > max_rows
                    rows = rows[:max_rows]
                    return QueryResult(columns=cols, rows=[list(r) for r in rows], truncated=truncated)
            except Exception as e:
                # 断连/超时等连接类错误：丢弃连接以便下次重连；SQL 本身的问题保留连接
                if self._is_connection_error(e):
                    self._close_locked()
                raise

    def query(self, sql: str) -> QueryResult:
        assert_read_only(sql)  # 防线 1：先于连接
        try:
            return self._run(sql)
        except SqlExecutionError:
            raise
        except Exception as e:  # 连接/执行失败 → 清晰错误
            raise SqlExecutionError(f"SQL 执行失败: {e}") from e

    def table_ddl(self, table_name: str) -> QueryResult:
        """表结构与索引：MySQL 模式下 SHOW CREATE TABLE 一次给全。

        表名只能拼进 SQL（query 契约不支持绑定参数），故先过标识符白名单。
        """
        safe = assert_safe_identifier(table_name, "表名")
        return self.query(f"show create table {safe}")

    def explain(self, sql: str) -> QueryResult:
        assert_read_only(sql)
        # TODO(联调): v1 走与 query 相同的执行路径，返回查询行而非执行计划（与 mock 的
        # explain 语义不同）。OB 上如需计划，对非 explain 开头的 sql 应改写为 "EXPLAIN {sql}"。
        try:
            return self._run(sql)
        except SqlExecutionError:
            raise
        except Exception as e:
            raise SqlExecutionError(f"EXPLAIN 失败: {e}") from e
