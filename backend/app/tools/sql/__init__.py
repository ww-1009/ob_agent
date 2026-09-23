"""只读 SQL 执行器。

- ``guard``：只读语法防线（白名单 / 禁止 FOR UPDATE / 标识符白名单）
- ``base``：真实执行器公共骨架 ``PooledSqlExecutor``（长连接 / 锁 / 只读 / 截断）
- ``mock``：基于 backend/data 夹具的离线执行器 ``MockSqlExecutor``（方言无关，MySQL 与 Oracle 租户通用）
- ``real``：PyMySQL 直连 MySQL 模式租户 ``MysqlSqlExecutor``
- ``oracle``：OCI 驱动（oracledb / cx_Oracle 可切换）连 Oracle 模式租户 ``OracleSqlExecutor``
- ``registry``：真实执行器长连接注册表，供应用关闭时统一释放

真实执行器统一按方言命名（``MysqlSqlExecutor`` / ``OracleSqlExecutor``，与 ``MockSqlExecutor``
同构）；旧名 ``RealSqlExecutor`` / ``OBOracleSqlExecutor`` 已随之重命名。

历史提示：本包曾有一个 `get_meta_db_executor` 工厂与 `meta_db` 配置段，两者都没有调用方，
已整体删除。如后续真需要接元数据库，从 git 历史取回即可。
"""
