"""应用日志初始化：把 root / uvicorn 的日志落盘（按大小轮转），控制台输出保留。

为什么需要这个模块
------------------

仓库里 13 个模块都用 ``logging.getLogger(__name__)`` 打日志，但全仓没有任何 handler
配置：root logger 没有 handler，于是

- ``logger.info`` / ``logger.debug`` 被**静默丢弃**；
- warning 及以上只靠 ``logging.lastResort`` 打到 stderr，**纯 message**（没有时间戳、
  级别、logger 名），终端一关就没了；
- ``app/agent/runner.py`` 的 ``logger.exception`` 堆栈留不住，而两个 README 都承诺
  「完整堆栈只留在服务端日志」。

uvicorn 自带的 ``LOGGING_CONFIG`` 只配置 ``uvicorn`` / ``uvicorn.error`` /
``uvicorn.access`` 这三个 logger（后两者 ``propagate=False`` 且各带自己的
StreamHandler），**不碰 root**。所以这里要做两件事：给 root 挂 handler（覆盖应用日志），
再把这同一个文件 handler **显式**挂到那两个 uvicorn logger 上（否则访问日志永远不落盘）。

这些 uvicorn logger 的**级别不归这里管**：``uvicorn.Config.configure_logging`` 会按
``--log-level``（默认 INFO）调 ``setLevel``，生产 unit 不传该参数即 INFO。因此
``logging.level`` 决定的是应用日志的详细程度，不要把 uvicorn 的 DEBUG 噪声拉进来的期望
放在它身上。

为什么单独一个模块、且不叫 ``app/logging.py``
---------------------------------------------

``app/logging.py`` 会遮蔽标准库 ``logging``（包内绝对导入会命中它），必须避开这个名字。
模块独立是因为它有多个入口：uvicorn 服务进程（``app/main.py``）、长时 CLI
（``app/agent/milvus_index.py``）以及测试。

为什么默认在 pytest 下不写文件
------------------------------

``app/main.py`` 文件末尾有模块级副作用 ``app = create_app()``，因此测试里一句
``from app.main import create_app`` 就会触发日志初始化。若不加守卫，跑一次 pytest 就会
往仓库写 ``backend/logs/app.log`` 并互相踩轮转。规则：检测到 pytest 就跳过，除非显式
设置 ``LOG_FILE``（明确的 opt-in）或 ``configure_logging(..., force=True)``。

已知限制
--------

``RotatingFileHandler`` **不是多进程安全的**（轮转时多个进程会互相覆盖/抢占）。
服务端已强制单 worker（``backend/run.sh`` 的 ``--workers 1``，HITL 确认通道、同 thread
串行锁、进程内单例 Milvus Lite 都依赖这一点）；需要多 worker 时应改走 journald
（systemd 默认）或按 pid 分文件。

失败哲学与检索层一致：**只降级不报错**——日志目录不可写时告警到 stderr，服务照常起。
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from app.config import LoggingConfig

_LOG_FORMAT = "%(asctime)s %(levelname)s [pid:%(process)d] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# uvicorn 的这两个 logger 都是 propagate=False 且自带 StreamHandler：
# 不给它们显式加 handler，启动行与访问日志就永远不会进文件。
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.access")

# 打开 root handler 后，这些库的 INFO 会灌满日志文件（pymilvus/grpc 的连接细节、
# httpx 的每条请求、jieba/faiss 的载入日志），默认压到 WARNING（可用 logging.third_party_level 调）。
_NOISY_LOGGERS = (
    "pymilvus",
    "milvus_lite",
    "grpc",
    "httpx",
    "httpcore",
    "urllib3",
    "jieba",
    "faiss",
)


class _NoiseFilter(logging.Filter):
    """按 logger 名 + 级别在**发这条路**上再压一次第三方噪声。

    只 ``setLevel`` 是不够的：jieba 在 import 时会把自己的 logger 重新设成 DEBUG
    （实测启动日志里混进 4 行 jieba DEBUG），faiss 也是 import 时才出现。handler 上的
    filter 拿到的始终是最终 record，不受库改自己级别影响。
    """

    def __init__(self, prefixes: tuple[str, ...], level: int) -> None:
        super().__init__()
        self._prefixes = prefixes
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= self._level:
            return True
        name = record.name
        return not any(name == p or name.startswith(p + ".") for p in self._prefixes)


# 本模块挂过的东西：重配/测试时要能干净摘掉，避免 handler 叠加导致日志重复。
_attached: list[tuple[logging.Logger, logging.Handler]] = []
_quiet_applied: list[tuple[logging.Logger, int]] = []
_configured_path: Path | None = None
_root_level_before: int | None = None


def _pytest_active() -> bool:
    """是否运行在 pytest 下（见模块 docstring 的「为什么默认不写文件」）。"""
    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def _level_of(value: str, fallback: int) -> int:
    """把级别名转成 int；无法识别时回退（合法性已由 config._validate_logging 把关）。"""
    mapping = logging.getLevelNamesMapping()
    return mapping.get(str(value).strip().upper(), fallback)


def reset_logging() -> None:
    """摘掉本模块挂过的 handler / 恢复被压级的 logger，回到「未初始化」状态。"""
    global _configured_path, _root_level_before
    for logger, handler in _attached:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响重配
            pass
    _attached.clear()
    for logger, level in _quiet_applied:
        logger.setLevel(level)
    _quiet_applied.clear()
    if _root_level_before is not None:
        logging.getLogger().setLevel(_root_level_before)
        _root_level_before = None
    _configured_path = None


def configure_logging(config: LoggingConfig | None = None, *, force: bool = False) -> Path | None:
    """初始化应用日志，返回实际启用的日志文件路径（未启用文件日志时返回 None）。

    - 幂等：已初始化过且 ``force=False`` 时直接返回既有路径（uvicorn ``--reload``、
      重复导入都不会叠加 handler）；
    - ``config=None`` 用 ``LoggingConfig()`` 默认值（配置解析不可用时的兜底）；
    - pytest 下默认跳过（见模块 docstring）；
    - 日志文件不可写时告警到 stderr 并继续（只降级不报错）。
    """
    global _configured_path, _root_level_before

    cfg = config if config is not None else LoggingConfig()
    if force:
        reset_logging()
    elif _configured_path is not None:
        return _configured_path

    if not force and _pytest_active() and not os.environ.get("LOG_FILE", "").strip():
        return None

    level = _level_of(cfg.level, logging.INFO)
    quiet_level = _level_of(cfg.third_party_level, logging.WARNING)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    noise_filter = _NoiseFilter(_NOISY_LOGGERS, quiet_level)
    root = logging.getLogger()
    if _root_level_before is None:
        _root_level_before = root.level
    root.setLevel(level)

    # 先挂控制台：这样后面「文件不可用」的告警本身也走统一格式，而不是 lastResort 的裸 message。
    if cfg.console:
        console = logging.StreamHandler()  # stderr：systemd 下仍进 journald
        console.setFormatter(formatter)
        console.addFilter(noise_filter)
        root.addHandler(console)
        _attached.append((root, console))

    path = cfg.resolve_file_path()
    handler: logging.Handler | None = None
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if cfg.max_bytes > 0:
                handler = logging.handlers.RotatingFileHandler(
                    path,
                    maxBytes=cfg.max_bytes,
                    backupCount=max(0, cfg.backups),
                    encoding="utf-8",
                    delay=True,  # 零记录时不建空文件
                )
            else:
                handler = logging.FileHandler(path, encoding="utf-8", delay=True)
            handler.setFormatter(formatter)
            handler.addFilter(noise_filter)
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "日志文件不可用（%s）：%s；仅保留控制台输出", path, exc
            )
            handler, path = None, None

    if handler is not None:
        root.addHandler(handler)
        _attached.append((root, handler))
        for name in _UVICORN_LOGGERS:
            uvicorn_logger = logging.getLogger(name)
            uvicorn_logger.addHandler(handler)
            _attached.append((uvicorn_logger, handler))

    for name in _NOISY_LOGGERS:
        noisy = logging.getLogger(name)
        _quiet_applied.append((noisy, noisy.level))
        noisy.setLevel(quiet_level)

    _configured_path = path
    logging.getLogger(__name__).info(
        "应用日志已初始化：file=%s level=%s console=%s", path or "(仅控制台)", cfg.level, cfg.console
    )
    return path