"""Real SQL 执行器：PyMySQL 直连，带只读三层防线。

三层防线：
1. 语法白名单（guard.assert_read_only），在任何连接之前执行。
2. 只读专用账号：要求在 dsn 中配置（配置文档建议 agent_ro 仅授 SELECT）。
3. 会话与执行防护：autocommit=1；超时；结果行数上限截断。

说明：连接/鉴权细节在真实环境就绪后联调确认（本任务离线只验证 1/缺 dsn 路径）。
"""
from __future__ import annotations

from urllib.parse import unquote, urlsplit

from app.config import SqlConfig
from app.tools.base import QueryResult, SqlExecutionError
from app.tools.sql.guard import assert_read_only

#
# def _parse_dsn(dsn: str) -> dict:
#     # dsn 形如 mysql+pymysql://user:pass@host:port/db（密码允许含 @ / 百分号编码）
#     try:
#         p = urlsplit(dsn)
#         port = p.port  # 非法端口抛 ValueError
#     except ValueError as e:
#         raise SqlExecutionError(f"无法解析 dsn: {dsn!r}") from e
#     if p.scheme != "mysql+pymysql" or not p.hostname or p.username is None:
#         raise SqlExecutionError(
#             f"无法解析 dsn: {dsn!r}（需形如 mysql+pymysql://user:pass@host:port/db）"
#         )
#     return {
#         "user": unquote(p.username),
#         "password": unquote(p.password or ""),
#         "host": p.hostname,
#         "port": port or 3306,
#         "db": p.path.lstrip("/") or None,
#     }


class RealSqlExecutor:
    def __init__(self, config: SqlConfig) -> None:
        self._cfg = config

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

    def _run(self, sql: str) -> QueryResult:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                max_rows = self._cfg.max_rows
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
                return QueryResult(columns=cols, rows=[list(r) for r in rows], truncated=truncated)
        finally:
            conn.close()

    def query(self, sql: str) -> QueryResult:
        assert_read_only(sql)  # 防线 1：先于连接
        try:
            return self._run(sql)
        except SqlExecutionError:
            raise
        except Exception as e:  # 连接/执行失败 → 清晰错误
            raise SqlExecutionError(f"SQL 执行失败: {e}") from e

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
