"""真实 SQL 执行器的长连接注册表。

MySQL 与 Oracle 执行器都会在构造时把自己登记进来，应用关闭时由 lifespan
统一 `close_all_executors()` 释放长连接。用弱引用集合，避免执行器随缓存淘汰
（或测试丢弃）后仍被注册表钉住内存。

放在独立模块而不是某个执行器文件里：Oracle 执行器不该为了关闭连接而反向
依赖 MySQL 执行器。
"""
from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型标注用，避免运行期循环导入
    from typing import Protocol

    class _Closable(Protocol):
        def close(self) -> None: ...

_LIVE_EXECUTORS: "weakref.WeakSet" = weakref.WeakSet()


def register(executor: "_Closable") -> None:
    """登记一个持有长连接的执行器（由执行器构造函数调用）。"""
    _LIVE_EXECUTORS.add(executor)


def close_all_executors() -> None:
    """关闭所有存活执行器的长连接；由应用 lifespan 的关闭钩子调用。"""
    for executor in list(_LIVE_EXECUTORS):
        try:
            executor.close()
        except Exception:  # noqa: BLE001 - 关闭失败不应阻断进程退出
            pass