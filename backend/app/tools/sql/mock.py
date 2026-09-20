"""Mock SQL 执行器：基于 backend/data 下的 JSON fixtures 模拟只读查询。

能力（v1 刻意最小）：
- SELECT <cols|*|count(*)> FROM <table> [WHERE col = 'value'] [LIMIT n]
  表名大小写不敏感；列清单可跨行；WHERE 仅支持单条等值条件
  （col = '字面量' 或 col = 裸值），值按字符串比较。
- FROM 命中 sample_tables.json 的表，或 oceanbase.gv$sql_audit（由 slow_sqls.json 生成）
- EXPLAIN <...> 单词匹配 explain_results.json 的 match 字段
其余语法（如 >、!=、AND 多条件、未知列/表）一律抛 SqlExecutionError。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.tools.base import QueryResult, SqlExecutionError
from app.tools.sql.guard import assert_read_only

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

_TABLE_TOKEN = re.compile(r"\bfrom\s+(?P<name>[`\w.$]+)", re.I)
_LIMIT = re.compile(r"\blimit\s+(\d+)", re.I)
# 等值条件：优先匹配带引号字面量，其次匹配裸值（如数字 id = 42）
_EQ_COND = re.compile(r"\b(\w+)\s*=\s*'([^']*)'|\b(\w+)\s*=\s*([^\s,;]+)", re.I)
_COUNT_STAR = re.compile(r"\bcount\s*\(\s*\*\s*\)", re.I)
_SELECT_COLS = re.compile(r"select\s+(.+?)\s+from\b", re.I | re.S)
_WHERE = re.compile(r"\bwhere\b", re.I)
_AND = re.compile(r"\band\b", re.I)


class MockSqlExecutor:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or _DEFAULT_DATA_DIR

    def query(self, sql: str) -> QueryResult:
        assert_read_only(sql)
        body = self._strip_prefix(sql, "explain")  # explain 交给 explain()
        if body is not None:
            return self.explain(body)
        table = self._table_of(sql)
        if table is None:
            raise SqlExecutionError("无法解析 FROM 表名")
        return self._select(sql, table)

    def explain(self, sql: str) -> QueryResult:
        assert_read_only(sql)
        return self._pick_explain(sql)

    # ---- 内部 ----

    def _pick_explain(self, sql: str) -> QueryResult:
        # 读取时允许通过 self._data_dir 覆盖默认路径
        p = self._data_dir / "explain_results.json"
        entries = json.loads(p.read_text(encoding="utf-8"))
        for e in entries:
            m = e.get("match")
            if m and re.search(rf"\b{re.escape(m)}\b", sql, re.I):
                return QueryResult(columns=e["columns"], rows=e["rows"])
        fallback = next((e for e in entries if e.get("match") == ""), None)
        if fallback is None:
            raise SqlExecutionError("explain_results.json 缺少兜底条目")
        return QueryResult(columns=fallback["columns"], rows=fallback["rows"])

    def _table_of(self, sql: str) -> str | None:
        m = _TABLE_TOKEN.search(sql)
        if not m:
            return None
        return m.group("name").split(".")[-1].strip("`")

    def _strip_prefix(self, sql: str, word: str) -> str | None:
        m = re.match(rf"^\s*{word}\b(.*)$", sql, re.I | re.S)
        return m.group(1).strip() if m else None

    def _select(self, sql: str, table: str) -> QueryResult:
        p = self._data_dir / "sample_tables.json"
        tables = json.loads(p.read_text(encoding="utf-8"))

        lookup = {k.lower(): v for k, v in tables.items()}

        if table.lower() == "gv$sql_audit":
            return self._select_audit(sql)

        if table.lower() not in lookup:
            raise SqlExecutionError(f"mock 中不存在表: {table}")

        data = lookup[table.lower()]
        columns = data["columns"]
        rows = list(data["rows"])

        # count(*)
        if _COUNT_STAR.search(sql):
            return QueryResult(columns=["count(*)"], rows=[[len(rows)]])

        # where 等值过滤（仅支持单条等值条件，其余 fail loud）
        cond = _EQ_COND.search(sql)
        if _WHERE.search(sql) and (cond is None or _AND.search(sql)):
            raise SqlExecutionError("mock 仅支持单条等值 WHERE 条件（col = value）")
        if cond:
            col = cond.group(1) if cond.group(1) is not None else cond.group(3)
            val = cond.group(2) if cond.group(2) is not None else cond.group(4)
            if col not in columns:
                raise SqlExecutionError(f"mock 不支持过滤列: {col}")
            rows = [r for r in rows if str(r[columns.index(col)]) == val]

        # limit
        lm = _LIMIT.search(sql)
        if lm:
            rows = rows[: int(lm.group(1))]

        # 目标列
        sel_m = _SELECT_COLS.search(sql)
        if sel_m:
            sel = sel_m.group(1).strip()
            if sel == "*":
                cols, out = columns, rows
            else:
                parts = [c.strip().strip("`") for c in sel.split(",")]
                if "count(*)" in sel:
                    raise SqlExecutionError("count 未解析，请写 count(*)")
                if any(c not in columns for c in parts):
                    raise SqlExecutionError(f"mock 不支持的列: {parts}")
                idx = [columns.index(c) for c in parts]
                cols, out = parts, [[r[i] for i in idx] for r in rows]
        else:
            cols, out = columns, rows

        return QueryResult(columns=cols, rows=out)

    def _select_audit(self, sql: str) -> QueryResult:
        p = self._data_dir / "slow_sqls.json"
        items = json.loads(p.read_text(encoding="utf-8"))
        rows = [
            [
                it["sql_id"],
                it["sql_text"],
                it["db_name"],
                it["avg_elapsed_us"],
                it["exec_count"],
            ]
            for it in items
        ]
        cols = ["sql_id", "sql_text", "db_name", "avg_elapsed_us", "exec_count"]
        lm = _LIMIT.search(sql)
        if lm:
            rows = rows[: int(lm.group(1))]
        return QueryResult(columns=cols, rows=rows)
