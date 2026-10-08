"""日志初始化（app.logging_setup）契约测试。

固化的是「日志真的落盘」这件事本身：文件 handler 与 uvicorn 挂载、幂等、级别、
轮转、pytest 守卫、以及「目录不可写只降级不报错」。全部在 tmp_path 下跑，force=True
显式绕过守卫，autouse fixture 负责摘干净 handler（否则 handler 会在用例间叠加，
既污染其他测试的 root logger，也会让「不重复」断言失效）。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from app import logging_setup as ls
from app.config import LoggingConfig


@pytest.fixture(autouse=True)
def _clean_logging():
    ls.reset_logging()
    yield
    ls.reset_logging()


def cfg_for(path: Path, **over) -> LoggingConfig:
    base = dict(file=str(path), level="INFO", console=True)
    base.update(over)
    return LoggingConfig(**base)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_configure_logging_writes_records_to_file(tmp_path):
    path = tmp_path / "app.log"
    assert ls.configure_logging(cfg_for(path), force=True) == path

    logging.getLogger("app.demo").info("落盘检查")

    text = read(path)
    # 时间戳 + 级别 + pid + logger 名 + 消息：lastResort 的裸 message 是没有这些的
    assert re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO \[pid:\d+\] app\.demo: 落盘检查$", text, re.M)
    assert "应用日志已初始化" in text


def test_configure_logging_attaches_file_handler_to_uvicorn_loggers(tmp_path):
    """uvicorn 与 uvicorn.access 都是 propagate=False，不显式挂就丢访问日志。

    级别不归我们管：``uvicorn.Config.configure_logging`` 会按 ``--log-level`` 调
    ``logging.getLogger("uvicorn.error").setLevel(...)``，而 ``tests/test_hitl_routing.py``
    在同一进程里起过 ``log_level="warning"`` 的真实 server，于是 ``uvicorn.error`` 被永久
    留在 WARNING，INFO 记录根本到不了 handler。所以这里显式调回 INFO 并在 finally 还原：
    本用例只断言「handler 挂对了」，uvicorn 的级别策略由下一个用例单独固化。
    """
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    levels = {name: logging.getLogger(name).level for name in names}
    for name in names:
        logging.getLogger(name).setLevel(logging.INFO)
    try:
        path = tmp_path / "app.log"
        ls.configure_logging(cfg_for(path), force=True)

        logging.getLogger("uvicorn").info("启动行")
        logging.getLogger("uvicorn.error").info("错误通道")
        logging.getLogger("uvicorn.access").info("GET /api/health 200")

        text = read(path)
        assert "启动行" in text          # uvicorn 自带 handler，直接落盘
        assert "错误通道" in text        # uvicorn.error 向父 uvicorn 传播
        assert "GET /api/health 200" in text
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


def test_configure_logging_leaves_uvicorn_logger_levels_alone(tmp_path):
    """刻意不替 uvicorn 决定级别（README 写明）：``--log-level`` 才是那个旋钮。"""
    uvicorn_error = logging.getLogger("uvicorn.error")
    before = uvicorn_error.level
    uvicorn_error.setLevel(logging.WARNING)
    try:
        ls.configure_logging(cfg_for(tmp_path / "app.log"), force=True)
        assert uvicorn_error.level == logging.WARNING
    finally:
        uvicorn_error.setLevel(before)


def test_configure_logging_is_idempotent(tmp_path):
    path = tmp_path / "app.log"
    root = logging.getLogger()
    before = {id(h) for h in root.handlers}

    cfg = cfg_for(path)
    assert ls.configure_logging(cfg, force=True) == path
    assert ls.configure_logging(cfg) == path      # 第二次直接复用

    # 只新增一个文件 handler（pytest 自己也会往 root 挂 _FileHandler，不能直接数）
    added = [h for h in root.handlers if id(h) not in before]
    assert len([h for h in added if isinstance(h, logging.FileHandler)]) == 1

    logging.getLogger("app.demo").info("只应出现一次")
    assert read(path).count("只应出现一次") == 1


def test_configure_logging_skips_under_pytest_without_force(tmp_path, monkeypatch):
    """app/main.py 的模块级 ``app = create_app()`` 让每次导入都调到这里，故默认不写文件。"""
    monkeypatch.delenv("LOG_FILE", raising=False)
    path = tmp_path / "app.log"

    assert ls.configure_logging(cfg_for(path)) is None
    assert not path.exists()


def test_configure_logging_force_reconfigures_to_new_file(tmp_path):
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    ls.configure_logging(cfg_for(first), force=True)
    ls.configure_logging(cfg_for(second), force=True)

    logging.getLogger("app.demo").info("只进 b")

    assert "只进 b" in read(second)
    assert "只进 b" not in read(first)


def test_rotate_when_max_bytes_exceeded(tmp_path):
    path = tmp_path / "app.log"
    ls.configure_logging(cfg_for(path, max_bytes=200, backups=2), force=True)

    logger = logging.getLogger("app.demo")
    for i in range(50):
        logger.info("填充第 %d 条：%s", i, "x" * 40)

    assert path.exists()
    assert (tmp_path / "app.log.1").exists()


def test_no_rotation_when_max_bytes_zero(tmp_path):
    path = tmp_path / "app.log"
    ls.configure_logging(cfg_for(path, max_bytes=0, backups=2), force=True)

    logger = logging.getLogger("app.demo")
    for i in range(50):
        logger.info("填充第 %d 条：%s", i, "x" * 40)

    assert path.exists()
    assert not (tmp_path / "app.log.1").exists()


def test_file_disabled_keeps_console_only(tmp_path):
    root = logging.getLogger()
    before = {id(h) for h in root.handlers}

    assert ls.configure_logging(cfg_for(tmp_path / "unused.log", file=""), force=True) is None

    added = [h for h in root.handlers if id(h) not in before]
    assert any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in added)
    assert not any(isinstance(h, logging.FileHandler) for h in added)
    assert not (tmp_path / "unused.log").exists()


def test_unwritable_log_dir_does_not_raise(tmp_path, capsys):
    """只降级不报错：日志目录建不出来时服务照常起（沿用检索降级哲学）。"""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    path = blocker / "sub" / "app.log"

    assert ls.configure_logging(cfg_for(path), force=True) is None
    assert not path.exists()
    assert "日志文件不可用" in capsys.readouterr().err


def test_quiet_third_party_loggers_are_capped(tmp_path):
    path = tmp_path / "app.log"
    ls.configure_logging(cfg_for(path), force=True)

    assert logging.getLogger("httpx").isEnabledFor(logging.INFO) is False
    assert logging.getLogger("pymilvus").isEnabledFor(logging.INFO) is False
    assert logging.getLogger("app.demo").isEnabledFor(logging.INFO) is True

    logging.getLogger("httpx").info("库噪声")
    logging.getLogger("app.demo").info("应用日志")

    text = read(path)
    assert "应用日志" in text
    assert "库噪声" not in text


def test_noise_filter_survives_library_resetting_its_level(tmp_path):
    """实测：jieba 在 import 时把自己的 logger 改回 DEBUG，所以光 setLevel 压不住，须靠 handler filter。"""
    path = tmp_path / "app.log"
    ls.configure_logging(cfg_for(path), force=True)

    jieba_logger = logging.getLogger("jieba")
    jieba_logger.setLevel(logging.DEBUG)          # 模拟库 import 时的自我配置
    jieba_logger.debug("jieba 的内部噪声")
    jieba_logger.warning("jieba 的真问题")        # 到 WARNING 仍要留下

    logging.getLogger("app.demo").info("应用日志")

    text = read(path)
    assert "jieba 的内部噪声" not in text
    assert "jieba 的真问题" in text
    assert "应用日志" in text


def test_traceback_is_persisted(tmp_path):
    """README 承诺「完整堆栈只留在服务端日志」，这里固化 runner.py 那种 logger.exception 的输出。"""
    path = tmp_path / "app.log"
    ls.configure_logging(cfg_for(path), force=True)

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:  # noqa: PERF203 - 故意模拟 runner 的 except 分支
        logging.getLogger("app.agent.runner").exception("agent 执行失败 [error_id=e1]: %s", exc)

    text = read(path)
    assert "agent 执行失败 [error_id=e1]: boom" in text
    assert "Traceback (most recent call last)" in text
    assert "RuntimeError: boom" in text


def test_reset_logging_removes_handlers_and_restores_levels(tmp_path):
    root = logging.getLogger()
    level_before = root.level
    before = {id(h) for h in root.handlers}

    ls.configure_logging(cfg_for(tmp_path / "app.log"), force=True)
    assert root.level == logging.INFO
    added = {id(h) for h in root.handlers} - before
    assert added   # 至少挂了控制台 handler，文件 handler 用延迟创建

    ls.reset_logging()

    assert root.level == level_before
    assert not (added & {id(h) for h in root.handlers})   # 本模块挂的全摘掉
    assert logging.getLogger("httpx").level == logging.NOTSET