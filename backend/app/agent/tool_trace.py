"""工具轨迹的构造与解析。

被 runner（正常工具调用结束）与 confirm（用户拒绝/审批超时）共用，
因此独立成模块，避免 runner ←→ confirm 的循环依赖。

事件不含任何行数据：只带工具名、入参摘要、耗时、成败、行数与是否被批准。
"""
from __future__ import annotations

import json
from typing import Mapping

from app.agent.tool_labels import tool_label

# 工具入参中单个字符串字段的最大长度（SQL/DDL 会很长，避免 SSE 与审计表膨胀）
ARG_MAX_CHARS = 500


def truncate_args(args: Mapping | None, limit: int = ARG_MAX_CHARS) -> dict:
    """入参截断：只截长字符串，其余类型原样保留。"""
    out: dict = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + "…"
        else:
            out[k] = v
    return out


def extract_tool_result(output: object) -> tuple[bool, str | None, int | None, bool | None]:
    """从工具返回解析 (ok, error, rows, truncated)。

    本项目工具统一返回 {"ok": ...} 的 JSON 字符串；文件工具返回纯文本，
    此时按「成功、无行数」处理。返回值里绝不含行数据。
    """
    content = getattr(output, "content", output)
    if not isinstance(content, str):
        return True, None, None, None
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return True, None, None, None
    if not isinstance(data, dict) or "ok" not in data:
        return True, None, None, None
    err = data.get("error")
    rows = data.get("row_count")
    if not isinstance(rows, int) and isinstance(data.get("items"), list):
        rows = len(data["items"])
    truncated = data.get("truncated")
    return (
        bool(data.get("ok")),
        str(err) if err else None,
        rows if isinstance(rows, int) else None,
        bool(truncated) if truncated is not None else None,
    )


def tool_trace_event(
    *,
    run_id: str,
    name: str,
    args: Mapping | None,
    ok: bool,
    error: str | None = None,
    rows: int | None = None,
    truncated: bool | None = None,
    approved: bool | None = None,
    duration_ms: int | None = None,
) -> dict:
    """工具轨迹事件（phase=end）。approved 只对受 HITL 管控的工具取值，其余为 None。"""
    return {
        "type": "tool",
        "phase": "end",
        "id": (run_id or "")[:8],
        "name": name,
        "label": tool_label(name),
        "args": truncate_args(args),
        "ok": bool(ok),
        "error": error,
        "rows": rows,
        "truncated": truncated,
        "approved": approved,
        "duration_ms": duration_ms,
    }
