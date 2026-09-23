"""只读 SQL 防线。

判定分两步：

1. **词法切分**：把 SQL 切成 token，注释、字符串字面量、引号标识符各自成类。
2. **在 token 序列上判定**：语句类型白名单 + 写关键字全量扫描 + 锁定读 / 导出文件 +
   多语句。

为什么要词法化：早期实现直接在原文上跑正则，行注释可以把关键字切开
（``select * from t for -- c\\n update`` 会被放行），而字符串里的 ``for update``
又会被误杀（``select 'for update'`` 被拒绝）。切成 token 后这两类问题同时消失。

判定原则是 **fail closed**：识别不了就拒绝。这里刻意不做完整语法解析——OceanBase
在标准 SQL 之外还有扩展语法，第三方解析器一旦不认就会误拒正常查询；而「词法化 +
写关键字全量扫描」已经能拦下核心风险：写语义不可能藏在注释、字符串或 CTE 后面溜过去。
"""
from __future__ import annotations

import re

from app.tools.base import SqlExecutionError

# 允许的语句首关键字
_ALLOWED_FIRST = {"select", "show", "explain", "describe", "desc", "with"}
# EXPLAIN 前缀里的修饰词：跳过它们才能拿到真正被执行的语句
_EXPLAIN_PREFIX = {"analyze", "plan", "for", "extended", "format", "into"}
# 出现即拒绝的写关键字。只匹配 token 序列，因此不会命中字符串 / 注释 / 引号标识符内部。
_WRITE_KEYWORDS = frozenset({
    "insert", "update", "delete", "merge", "upsert", "replace",
    "drop", "alter", "truncate", "create", "grant", "revoke", "rename",
    "call", "execute", "purge", "flashback", "load", "lock", "unlock",
    "outfile", "dumpfile", "handler",
    "begin", "commit", "rollback", "savepoint", "prepare", "deallocate",
    "kill", "shutdown", "flush", "install", "uninstall", "optimize", "repair",
})
# 这些语句本身只读，且合法文本里就会包含写关键字（如 `SHOW CREATE TABLE`），跳过扫描
_SKIP_WRITE_SCAN = {"show", "describe", "desc"}
# 表名/标识符：允许可选的 schema 前缀，字符集取 OceanBase MySQL 与 Oracle 的并集
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$#]*(?:\.[A-Za-z_][A-Za-z0-9_$#]*)?$")
# 标识符里额外允许出现的字符（词法切分用）
_WORD_CHARS = "_$#"


class ReadOnlyViolation(SqlExecutionError):
    """SQL 违反只读约束。"""


def _skip_line(sql: str, i: int) -> int:
    end = sql.find("\n", i)
    return len(sql) if end < 0 else end + 1


def _read_quoted(sql: str, start: int) -> tuple[str, int]:
    """读到成对的引号，返回（内容, 下一位置）。未闭合时读到结尾（等价于容错，fail closed）。"""
    quote = sql[start]
    i, n = start + 1, len(sql)
    buf: list[str] = []
    while i < n:
        ch = sql[i]
        if ch == "\\" and quote == "'" and i + 1 < n:  # 反斜杠转义只对 MySQL 字符串
            buf.append(sql[i + 1])
            i += 2
            continue
        if ch == quote:
            if i + 1 < n and sql[i + 1] == quote:  # '' / "" / `` 双写转义
                buf.append(quote)
                i += 2
                continue
            i += 1
            break
        buf.append(ch)
        i += 1
    return "".join(buf), i


def _tokenize(sql: str) -> list[tuple[str, str]]:
    """切成 (kind, text)，kind ∈ {word, string, ident, punct}；空白与注释被丢弃。"""
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        # 行注释：MySQL 要求 `--` 后跟空白；只有在确认是注释时才丢弃，否则留在 token
        # 流里按普通字符扫描（宁可多拒，也不给「注释藏关键字」留口子）。
        if ch == "-" and sql.startswith("--", i) and (i + 2 >= n or sql[i + 2].isspace()):
            i = _skip_line(sql, i)
            continue
        if ch == "#":
            i = _skip_line(sql, i)
            continue
        if ch == "/" and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch in "'\"`":
            text, i = _read_quoted(sql, i)
            tokens.append(("string" if ch == "'" else "ident", text))
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (sql[j].isalnum() or sql[j] in _WORD_CHARS):
                j += 1
            tokens.append(("word", sql[i:j]))
            i = j
            continue
        tokens.append(("punct", ch))
        i += 1
    return tokens


def _words(tokens: list[tuple[str, str]]) -> list[str]:
    return [text.lower() for kind, text in tokens if kind == "word"]


def _first_keyword(tokens: list[tuple[str, str]]) -> str:
    if tokens and tokens[0][0] == "word":
        return tokens[0][1].lower()
    return ""


def _inner_statement_keyword(tokens: list[tuple[str, str]]) -> str:
    """跳过 EXPLAIN 前缀，返回真正被执行的语句首关键字（识别不出返回空串）。"""
    i, n = 1, len(tokens)
    while i < n:
        kind, text = tokens[i]
        low = text.lower()
        if kind == "word" and low in _EXPLAIN_PREFIX:
            i += 1
            if low == "format":  # FORMAT = TREE / FORMAT JSON
                if i < n and tokens[i] == ("punct", "="):
                    i += 1
                if i < n and tokens[i][0] == "word":
                    i += 1
            continue
        break
    if i < n and tokens[i][0] == "word":
        return tokens[i][1].lower()
    return ""


def _has_sequence(words: list[str], *seq: str) -> bool:
    size = len(seq)
    if not size or len(words) < size:
        return False
    target = list(seq)
    return any(words[i:i + size] == target for i in range(len(words) - size + 1))


def assert_read_only(sql: str) -> None:
    tokens = _tokenize(sql)
    # 尾部空语句允许（`select 1;` 与 `select 1` 等价）
    while tokens and tokens[-1] == ("punct", ";"):
        tokens.pop()
    if not tokens:
        raise ReadOnlyViolation("空 SQL")

    # 多语句：此刻还出现分号，说明分号后面还跟着语句
    if any(token == ("punct", ";") for token in tokens):
        raise ReadOnlyViolation("多语句/带分号的额外语句不被允许")

    first = _first_keyword(tokens)
    if first not in _ALLOWED_FIRST:
        raise ReadOnlyViolation(f"只读模式禁止语句类型: {first or '<无法识别>'}")

    # EXPLAIN 必须看内层语句：旧实现整体跳过黑名单，`explain analyze delete from t` 被放行
    statement = _inner_statement_keyword(tokens) if first == "explain" else first
    if statement not in _ALLOWED_FIRST or statement == "explain":
        raise ReadOnlyViolation(f"只读模式禁止语句类型: {statement or '<无法识别>'}")

    words = _words(tokens)
    if (
        _has_sequence(words, "for", "update")
        or _has_sequence(words, "for", "share")
        or _has_sequence(words, "lock", "in", "share", "mode")
    ):
        raise ReadOnlyViolation("SELECT ... FOR UPDATE / LOCK IN SHARE MODE 不被允许")

    if statement not in _SKIP_WRITE_SCAN:
        # CTE 之后的 DML（`with x as (...) delete from t`）在这条全覆盖扫描里现形
        hit = next((word for word in words if word in _WRITE_KEYWORDS), "")
        if hit:
            raise ReadOnlyViolation(f"只读模式禁止语句包含写关键字: {hit}")


def assert_safe_identifier(name: str, kind: str = "标识符") -> str:
    """校验表名等标识符并返回去掉引号包裹的形式。

    表结构类工具（show create table / dbms_metadata.get_ddl）只能把表名拼进 SQL
    字符串（执行器的 query(sql) 契约不支持绑定参数），所以这里用严格白名单挡住拼接
    注入。允许 `schema.name` 形式，字符集取 MySQL 与 Oracle 的并集。
    """
    n = (name or "").strip().strip("`\"")
    if not _IDENTIFIER.match(n):
        raise ReadOnlyViolation(
            f"{kind}不合法: {name!r}（只允许字母/数字/_/$/#，可带一个 schema 前缀）"
        )
    return n