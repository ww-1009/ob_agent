"""只读工具调用的审计留痕（与 chat_message、检查点共用同一个 PG 连接池）。

记录「谁在哪个会话、对哪个工具、用什么入参、耗时多久、成功还是失败、是否被人工批准」。
绝不记录查询结果行数据；入参由 runner 侧截断后再传入。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

SCHEMA_SQL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS audit_event (
      id          BIGSERIAL PRIMARY KEY,
      thread_id   TEXT,
      tool        TEXT NOT NULL,
      args        JSONB NOT NULL DEFAULT '{}'::jsonb,
      ok          BOOLEAN NOT NULL,
      error       TEXT,
      rows        INTEGER,
      truncated   BOOLEAN,
      approved    BOOLEAN,
      duration_ms INTEGER,
      created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS audit_event_id_idx ON audit_event (id DESC)",
    "CREATE INDEX IF NOT EXISTS audit_event_thread_idx ON audit_event (thread_id, id)",
    "CREATE INDEX IF NOT EXISTS audit_event_tool_idx ON audit_event (tool, id DESC)",
)

_COLUMNS = (
    "id, thread_id, tool, args, ok, error, rows, truncated, approved, duration_ms, created_at"
)


@dataclass
class AuditEvent:
    id: int
    thread_id: Optional[str]
    tool: str
    args: dict
    ok: bool
    error: Optional[str]
    rows: Optional[int]
    truncated: Optional[bool]
    approved: Optional[bool]
    duration_ms: Optional[int]
    created_at: datetime


class AuditStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for stmt in SCHEMA_SQL:
                    await cur.execute(stmt)

    async def append(
        self,
        *,
        thread_id: Optional[str],
        tool: str,
        args: Optional[dict],
        ok: bool,
        error: Optional[str] = None,
        rows: Optional[int] = None,
        truncated: Optional[bool] = None,
        approved: Optional[bool] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        sql = """
            INSERT INTO audit_event
              (thread_id, tool, args, ok, error, rows, truncated, approved, duration_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (
                    thread_id, tool, Jsonb(args or {}), bool(ok), error,
                    rows, truncated, approved, duration_ms,
                ))

    async def list_events(
        self,
        limit: int,
        *,
        thread_id: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> list[AuditEvent]:
        where: list[str] = []
        params: list[Any] = []
        if thread_id:
            where.append("thread_id = %s")
            params.append(thread_id)
        if tool:
            where.append("tool = %s")
            params.append(tool)
        sql = f"SELECT {_COLUMNS} FROM audit_event"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT %s"
        params.append(int(limit))

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
        return [
            AuditEvent(
                id=r["id"],
                thread_id=r["thread_id"],
                tool=r["tool"],
                args=r["args"] or {},
                ok=r["ok"],
                error=r["error"],
                rows=r["rows"],
                truncated=r["truncated"],
                approved=r["approved"],
                duration_ms=r["duration_ms"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
