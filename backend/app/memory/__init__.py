"""会话记忆运行时：LangGraph Postgres 检查点 + 会话历史表 + 工具审计表。

一个 AsyncConnectionPool 同时供 AsyncPostgresSaver、ChatMessageStore 与 AuditStore 使用。
连接参数必须对齐上游 from_conn_string 的设置（autocommit / prepare_threshold / dict_row）：
检查点内部按 dict 取值（value["thread_id"]），row_factory 不对会直接 KeyError/TypeError。

打开失败会重试若干次；最终失败则降级为「记忆关闭」并返回失败原因（供 /api/health 展示），
不阻断服务启动。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import MemoryConfig
from app.memory.audit import AuditStore
from app.memory.store import ChatMessageStore

if TYPE_CHECKING:  # 仅类型标注：运行期按需导入，未安装依赖时本模块仍可导入
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)

# 上游 setup() 建的表；多 worker 并发 setup 冲突时用它判定「其实已经就绪」
_REQUIRED_TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")

# 启动期重试：PG 偶发抖动（容器刚起来、连接被拒）不应让整个进程永久失去记忆与审计。
# 次数与单次等待上限由配置控制，保证最坏情况的启动阻塞有界（见 MemoryConfig）。
_RETRY_BASE_DELAY = 0.5


@dataclass
class MemoryRuntime:
    pool: AsyncConnectionPool
    checkpointer: "AsyncPostgresSaver"
    store: ChatMessageStore
    audit: AuditStore

    async def aclose(self) -> None:
        await self.pool.close()

    async def adelete_thread(self, thread_id: str) -> int:
        """删除会话：历史表行 + 检查点三表（避免残留）。返回删除的历史消息数。

        注意：审计行**不**随之删除（审计只增不改，见 memory/audit.py）。
        """
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


async def _open_once(cfg: MemoryConfig, dsn: str) -> tuple[Optional[MemoryRuntime], Optional[str]]:
    """尝试打开一次；失败关闭已建资源并返回原因。"""
    # AsyncConnectionPool 的构造也会校验参数（min_size > max_size → ValueError）。
    # 构造必须包在 try 内：异常若穿透 open_memory → lifespan 就违背了本模块
    # 「打开失败降级为记忆关闭、不阻断服务启动」的契约（见模块 docstring）。
    try:
        pool = AsyncConnectionPool(
            dsn,
            min_size=cfg.pool_min_size,
            max_size=cfg.pool_max_size,
            open=False,
            # check：取连接时校验并自动重连——PG 重启/空闲被服务端断开后，池里的死连接
            # 否则会在第一次使用时报错（max_idle/max_lifetime 只保证最终回收）
            check=AsyncConnectionPool.check_connection,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
    except Exception as e:
        return None, f"连接池参数错误: {e}"
    try:
        await pool.open(wait=True, timeout=cfg.open_timeout_seconds)
    except Exception as e:
        await _close_quietly(pool)
        return None, f"连接 PG 失败: {e}"

    try:
        store = ChatMessageStore(pool)
        await store.ensure_schema()
        audit = AuditStore(pool)
        await audit.ensure_schema()
        if cfg.audit_retention_days > 0:
            pruned = await audit.prune(cfg.audit_retention_days)
            logger.info("审计保留期 %d 天：启动清理 %d 行", cfg.audit_retention_days, pruned)

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
        await _close_quietly(pool)
        return None, f"建表/迁移失败: {e}"

    logger.info("会话记忆已启用（thread 级检查点 + chat_message 历史表 + audit_event 审计表）")
    return MemoryRuntime(pool=pool, checkpointer=saver, store=store, audit=audit), None


async def _close_quietly(pool: AsyncConnectionPool) -> None:
    try:
        await pool.close()
    except Exception:  # noqa: BLE001 - 关闭失败不影响降级
        pass


async def open_memory(cfg: MemoryConfig) -> tuple[Optional[MemoryRuntime], Optional[str]]:
    """打开记忆运行时。

    返回 (runtime, error)：成功为 (runtime, None)，最终失败为 (None, 原因)。
    原因字符串会被 /api/health 暴露为 memory_error，便于线上快速定位降级原因。
    """
    try:
        dsn = cfg.build_dsn()
    except Exception as e:
        logger.warning("记忆初始化失败（配置）：%s", e)
        return None, f"配置错误: {e}"

    # 非法池参数属确定性配置错误，重试无意义：直接降级并给出明确原因。
    if cfg.pool_min_size > cfg.pool_max_size:
        msg = f"pool_min_size({cfg.pool_min_size}) 不能大于 pool_max_size({cfg.pool_max_size})"
        logger.warning("记忆初始化失败（配置）：%s", msg)
        return None, f"配置错误: {msg}"

    last_error: Optional[str] = None
    attempts = max(1, cfg.open_attempts)
    for attempt in range(1, attempts + 1):
        runtime, error = await _open_once(cfg, dsn)
        if runtime is not None:
            return runtime, None
        last_error = error
        if attempt < attempts:
            delay = _RETRY_BASE_DELAY * attempt
            logger.warning("记忆初始化第 %d/%d 次失败（%s）；%.1fs 后重试", attempt, attempts, error, delay)
            await asyncio.sleep(delay)

    logger.warning("记忆初始化最终失败：%s；本次以「无记忆」模式运行", last_error)
    return None, last_error
