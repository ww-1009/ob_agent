"""只读 SQL 防线。

规则：
1. 首关键字白名单：select/show/explain/describe/desc/with。
2. select 后禁止 for update / lock in share mode（粗匹配，足够 v1）。
3. select 后禁止 into outfile/dumpfile（防止落盘写文件）。
4. 禁止多语句（除尾部一个分号外出现分号后跟语句即拒绝）。
注释（-- 开头行 / /* */）会被剥离后判断。
"""
from __future__ import annotations

import re

from app.tools.base import SqlExecutionError

_ALLOWED_FIRST = {"select", "show", "explain", "describe", "desc", "with"}
_LEADING_COMMENT = re.compile(r"^\s*(--[^\n]*\n?|/\*.*?\*/|#[^\n]*\n?)*", re.S)
_FOR_UPDATE = re.compile(
    r"\bfor(?:\s|/\*.*?\*/)*update\b"
    r"|\block(?:\s|/\*.*?\*/)*in(?:\s|/\*.*?\*/)*share(?:\s|/\*.*?\*/)*mode\b",
    re.I | re.S,
)
_INTO_FILE = re.compile(r"\binto(?:\s|/\*.*?\*/)*(?:outfile|dumpfile)\b", re.I | re.S)
_MULTI_STMT = re.compile(r";\s*\S")


class ReadOnlyViolation(SqlExecutionError):
    """SQL 违反只读约束。"""


def _clean(sql: str) -> str:
    # 去掉首部注释
    cleaned = _LEADING_COMMENT.sub("", sql)
    # 去掉整体包裹的空白与尾部空语句
    cleaned = cleaned.strip().rstrip(";").strip()
    return cleaned


def _first_keyword(sql: str) -> str:
    m = re.match(r"[A-Za-z]+", sql)
    return m.group(0).lower() if m else ""


def assert_read_only(sql: str) -> None:
    cleaned = _clean(sql)
    if not cleaned:
        raise ReadOnlyViolation("空 SQL")
    kw = _first_keyword(cleaned)
    if kw not in _ALLOWED_FIRST:
        raise ReadOnlyViolation(f"只读模式禁止语句类型: {kw or '<无法识别>'}")

    # with 开头的 CTE 其后必须是 select 主句，此处不深入解析（v1 简化），靠账号兜底。
    if kw in {"select", "with"}:
        if _FOR_UPDATE.search(cleaned):
            raise ReadOnlyViolation("SELECT ... FOR UPDATE 不被允许")
        if _INTO_FILE.search(cleaned):
            raise ReadOnlyViolation("SELECT ... INTO OUTFILE/DUMPFILE 不被允许（只读模式禁止写服务器文件）")

    # 多语句检查
    if _MULTI_STMT.search(cleaned):
        raise ReadOnlyViolation("多语句/带分号的额外语句不被允许")
