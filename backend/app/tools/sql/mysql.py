"""MySQL 模式租户的真实 SQL 执行器：PyMySQL 直连。

只读三层防线（公共骨架见 ``app.tools.sql.base``）：
1. 语法白名单（guard.assert_read_only），在任何连接之前执行。
2. 只读专用账号：建议 sql_ro 配置仅授 SELECT 的账号。
3. 会话与执行防护：autocommit=1；超时；结果行数上限截断。

连接按执行器实例复用（长连接 + ping(reconnect=True)）：早先每次查询都新建并关闭连接，
一轮对话多次调用工具就要多次 TCP + 鉴权（经 obproxy 时延迟明显）。
"""
from __future__ import annotations

from dataclasses import replace

from app.config import SqlConfig
from app.tools.base import QueryResult, SqlExecutionError
from app.tools.sql.base import PooledSqlExecutor
from app.tools.sql.guard import assert_safe_identifier

# 长连接注册表（MySQL 与 Oracle 执行器共用）。close_all_executors 在此顺带再导出，
# 方便按方言模块取用；应用装配请直接从 app.tools.sql.registry 导入。
from app.tools.sql.registry import close_all_executors  # noqa: F401

__all__ = ["MysqlSqlExecutor", "resolve_config", "close_all_executors"]


def resolve_config(cfg: SqlConfig, tenant_name: str, cluster_name: str) -> SqlConfig:
    """把 sql_ro 的通用配置解析成「本租户可直连」的 MySQL 配置。

    MySQL 模式的租户与集群写在用户名里：``user@tenant#cluster``（Oracle 模式则把
    工具传入的 db_name 当作 DSN 的 service name，不写进用户名）。配置里已含 ``@``
    就原样用，便于显式指定租户。
    """
    base = (cfg.username or "").strip()
    username = base if "@" in base else f"{base}@{tenant_name}#{cluster_name}"
    return replace(cfg, username=username)


class MysqlSqlExecutor(PooledSqlExecutor):
    """PyMySQL 执行器；连接生命周期与只读语义全部继承自 ``PooledSqlExecutor``。"""

    def _load_driver(self):
        try:
            import pymysql
        except ImportError as e:  # pragma: no cover - 取决于部署环境
            raise SqlExecutionError("缺少依赖 pymysql，请安装") from e
        return pymysql

    def _connect(self):
        drv = self._driver()
        # TODO(联调): 大结果集可切 SSCursor 服务端游标逐批拉取（现为客户端 fetchmany+截断）；
        # 服务端超时 hint（ob_query_timeout）与客户端 read_timeout 的配合待联调确认。
        return drv.connect(
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

    def _ping(self, conn) -> None:
        conn.ping(reconnect=True)  # PyMySQL 支持原地重连，不必丢弃连接

    def _is_connection_error(self, exc: BaseException) -> bool:
        """只有连接类错误才丢弃连接；SQL 语法/权限等错误保留连接。"""
        drv = self._driver()
        return isinstance(exc, (drv.err.OperationalError, drv.err.InterfaceError))

    def table_ddl(self, table_name: str) -> QueryResult:
        """表结构与索引：MySQL 模式下 SHOW CREATE TABLE 一次给全。

        表名只能拼进 SQL（query 契约不支持绑定参数），故先过标识符白名单。
        """
        safe = assert_safe_identifier(table_name, "表名")
        return self.query(f"show create table {safe}")