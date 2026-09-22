"""会话历史的展示投影存储（自建表，与 LangGraph 检查点解耦）。

为什么不用检查点直接渲染历史：agent 挂了 SummarizationMiddleware，上下文达阈值时会
用摘要改写 state 里的 messages，早期消息会从检查点中消失。因此检查点只作为「LLM 记忆」
真源，本表保存 user/assistant 终稿，保证前端历史完整。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from psycopg_pool import AsyncConnectionPool

SCHEMA_SQL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS chat_message (
      id         BIGSERIAL PRIMARY KEY,
      thread_id  TEXT NOT NULL,
      role       TEXT NOT NULL CHECK (role IN ('user','assistant')),
      content    TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS chat_message_thread_idx ON chat_message (thread_id, id)",
)

_WS = re.compile(r"\s+")
TITLE_MAX = 40
DEFAULT_TITLE = "新对话"


def thread_title(first_user_content: Optional[str]) -> str:
    """会话标题：首条用户提问折叠空白后截断；无则取默认名。"""
    if not first_user_content:
        return DEFAULT_TITLE
    collapsed = _WS.sub(" ", first_user_content).strip()
    if not collapsed:
        return DEFAULT_TITLE
    return collapsed[:TITLE_MAX]


@dataclass
class StoredMessage:
    role: str
    content: str
    created_at: datetime


@dataclass
class ThreadSummary:
    thread_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ChatMessageStore:
    """基于 psycopg AsyncConnectionPool 的会话历史读写（连接已 autocommit + dict_row）。"""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for stmt in SCHEMA_SQL:
                    await cur.execute(stmt)

    async def append(self, thread_id: str, role: str, content: str) -> None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO chat_message (thread_id, role, content) VALUES (%s, %s, %s)",
                    (thread_id, role, content),
                )

    async def list_threads(self, limit: int) -> list[ThreadSummary]:
        sql = """
            SELECT m.thread_id                          AS thread_id,
                   MIN(m.created_at)                    AS created_at,
                   MAX(m.created_at)                    AS updated_at,
                   COUNT(*)::int                        AS message_count,
                   (SELECT u.content
                      FROM chat_message u
                     WHERE u.thread_id = m.thread_id AND u.role = 'user'
                     ORDER BY u.id
                     LIMIT 1)                           AS first_user_content
              FROM chat_message m
             GROUP BY m.thread_id
             ORDER BY updated_at DESC
             LIMIT %s
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (int(limit),))
                rows = await cur.fetchall()
        return [
            ThreadSummary(
                thread_id=r["thread_id"],
                title=thread_title(r["first_user_content"]),
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                message_count=r["message_count"],
            )
            for r in rows
        ]

    async def get_messages(self, thread_id: str, limit: int) -> list[StoredMessage]:
        # 内层倒序取最近 limit 条，外层再正序，保证返回时间正序
        sql = """
            SELECT role, content, created_at FROM (
              SELECT id, role, content, created_at
                FROM chat_message
               WHERE thread_id = %s
               ORDER BY id DESC
               LIMIT %s
            ) t ORDER BY id ASC
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (thread_id, int(limit)))
                rows = await cur.fetchall()
        return [StoredMessage(role=r["role"], content=r["content"], created_at=r["created_at"]) for r in rows]

    async def delete_thread(self, thread_id: str) -> int:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM chat_message WHERE thread_id = %s", (thread_id,))
                return int(cur.rowcount or 0)
