"""SQL 执行器工厂。

注意：get_meta_db_executor 目前没有调用方——meta_db 配置段在当前版本尚未使用
（README 已注明）。保留它是为后续 meta_db 能力接入预留的装配位。
"""
from __future__ import annotations

from app.config import Settings
from app.tools.sql.mock import MockSqlExecutor


def get_meta_db_executor(settings: Settings):
    if settings.meta_db.provider == "real":
        from app.tools.sql.real import RealSqlExecutor  # Task 8 实现

        return RealSqlExecutor(settings.meta_db)
    return MockSqlExecutor()
