"""会话记忆运行时：LangGraph Postgres 检查点 + 会话历史表。

一个 AsyncConnectionPool 同时供 AsyncPostgresSaver 与 ChatMessageStore 使用。
连接参数必须对齐上游 from_conn_string 的设置（autocommit / prepare_threshold / dict_row）：
检查点内部按 dict 取值（value["thread_id"]），row_factory 不对会直接 KeyError/TypeError。

任何打开失败都降级为「记忆关闭」，不阻断服务启动（见 open_memory）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import MemoryConfig
from app.memory.store import ChatMessageStore

if TYPE_CHECKING:  # 仅类型标注：运行期按需导入，未安装依赖时本模块仍可导入
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

# 上游 setup() 建的表；多 worker 并发 setup 冲突时用它判定「其实已经就绪」
_REQUIRED_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


@dataclass
class MemoryRuntime:
    pool: AsyncConnectionPool
    checkpointer: "AsyncPostgresSaver"
    store: ChatMessageStore

    async def aclose(self) -> None:
        await self.pool.close()

    async def adelete_thread(self, thread_id: str) -> int:
        """删除会话：历史表行 + 检查点三表（避免残留）。返回删除的历史消息数。"""
        deleted = await self.store.delete_thread(thread_id)
        await self.checkpointer.adelete_thread(thread_id)
        return deleted


async def _tables_ready(pool: AsyncConnectionPool) -> bool:
    """检查点所需表是否齐备（setup() 报错但表已在时视为就绪）。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            for table in _REQUIRED_TABLES:
                await cur.execute("SELECT to_regclass(%s) IS NOT NULL AS ok", (f"public.{table}",))
                row = await cur.fetchone()
                if not (row and row["ok"]):
                    return False
    return True


async def open_memory(cfg: MemoryConfig) -> Optional[MemoryRuntime]:
    """打开记忆运行时；失败返回 None（调用方退回无状态模式）。"""
    try:
        dsn = cfg.build_dsn()
    except Exception as e:
        logger.warning("记忆初始化失败（配置）：%s", e)
        return None

    pool = AsyncConnectionPool(
        dsn,
        min_size=cfg.pool_min_size,
        max_size=cfg.pool_max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        await pool.open(wait=True, timeout=10)
    except Exception as e:
        logger.warning("记忆初始化失败（连接 PG）：%s", e)
        try:
            await pool.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响降级
            pass
        return None

    store = ChatMessageStore(pool)
    try:
        await store.ensure_schema()
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # 必须在运行中的事件循环里构造（内部会取 running loop）
        saver = AsyncPostgresSaver(pool)
        try:
            await saver.setup()
        except Exception as e:  # 多 worker 并发迁移可能主键冲突
            if await _tables_ready(pool):
                logger.info("检查点 setup() 报错但表已就绪，继续使用：%s", e)
            else:
                raise
    except Exception as e:
        logger.warning("记忆初始化失败（建表/迁移）：%s；本次以「无记忆」模式运行", e)
        try:
            await pool.close()
        except Exception:  # noqa: BLE001
            pass
        return None

    logger.info("会话记忆已启用（thread 级检查点 + chat_message 历史表）")
    return MemoryRuntime(pool=pool, checkpointer=saver, store=store)
