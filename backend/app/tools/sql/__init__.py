"""SQL 执行器工厂。"""
from __future__ import annotations

from app.config import Settings
from app.tools.sql.mock import MockSqlExecutor


def get_meta_db_executor(settings: Settings):
    if settings.meta_db.provider == "real":
        from app.tools.sql.real import RealSqlExecutor  # Task 8 实现

        return RealSqlExecutor(settings.meta_db)
    return MockSqlExecutor()
