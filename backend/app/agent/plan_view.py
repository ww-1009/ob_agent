"""执行计划的可视化视图（服务端归一化，前端只负责渲染）。

OCP 的 explain 报文是一张「节点表 + children 引用」的图::

    {"data": [{"id": 1, "operator": "EXCHANGE OUT", "name": "distributed",
               "rows": 1280, "cost": 48230, "property": "...", "children": [0]}],
     "planOperationSummery": [{"operator": "TABLE SCAN", "count": 1, ...}],
     "rootOperations": [1]}

直接丢给前端，前端要自己重建树、还要防环；mock fixture 与真实报文的键名差异也得各写一遍。
这里一次性归一成「先序 + depth」的扁平列表与算子汇总，前端只做渲染。

只在 get_sql_explain 成功时挂到 tool 事件上，且不含任何行数据：算子名、估算行数、代价都是
计划元信息。节点数有上限，避免一份巨大计划把 SSE 与前端拖垮。
"""
from __future__ import annotations

import json
from typing import Any

# 唯一会带计划视图的工具
PLAN_TOOL = "get_sql_explain"

_MAX_NODES = 200
_MAX_SUMMARY_ROWS = 50
_MAX_TEXT_CHARS = 80
_MAX_PROPERTY_CHARS = 240


def _text(value: Any, limit: int = _MAX_TEXT_CHARS) -> str:
    if value is None:
        return ""
    s = str(value)
    return s[:limit] + "…" if len(s) > limit else s


def _int(value: Any) -> int | None:
    """宽松取整：OCP 有的字段是 "1280" 这样的字符串，取不到就给 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _pick(payload: dict, *keys: str) -> Any:
    """按序取首个非 None 的键：兼容 OCP 原始驼峰与工具返回里的下划线命名。"""
    for key in keys:
        if payload.get(key) is not None:
            return payload[key]
    return None


def _index_nodes(raw: list) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """登记所有节点，返回（节点表, 每个节点的子键列表）。

    节点 id 缺失时用位置序号兜底；children 里既可能是子节点 id（OCP 实际报文），
    也可能是内联的子节点对象（联调期见过的变体），两种都收。
    """
    nodes: dict[str, dict] = {}
    children: dict[str, list[str]] = {}

    def register(item: Any, fallback: str) -> str | None:
        if not isinstance(item, dict):
            return None
        key = str(item["id"]) if item.get("id") is not None else fallback
        if key in nodes:  # 重复引用：第一次为准
            return key
        nodes[key] = item
        kids: list[str] = []
        for idx, child in enumerate(item.get("children") or []):
            if isinstance(child, dict):
                sub = register(child, f"{key}.{idx}")
                if sub is not None:
                    kids.append(sub)
            else:
                kids.append(str(child))
        children[key] = kids
        return key

    for i, item in enumerate(raw):
        register(item, f"node-{i}")
    return nodes, children


def _roots(nodes: dict[str, dict], children: dict[str, list[str]], declared: Any) -> list[str]:
    """根节点：优先用报文声明的 rootOperations，否则取没有父节点的那些。"""
    declared_keys = []
    if isinstance(declared, list):
        declared_keys = [str(r) for r in declared if str(r) in nodes]
    if declared_keys:
        return declared_keys
    parent_of: set[str] = set()
    for kids in children.values():
        parent_of.update(k for k in kids if k in nodes)
    return [key for key in nodes if key not in parent_of]


def _walk(nodes: dict[str, dict], children: dict[str, list[str]], roots: list[str]) -> list[tuple[str, int]]:
    """先序 + depth。visited 集合同时挡住环与重复引用，不会无限递归。"""
    seen: set[str] = set()
    out: list[tuple[str, int]] = []

    def visit(key: str, depth: int) -> None:
        if key in seen or key not in nodes:
            return
        seen.add(key)
        out.append((key, depth))
        for kid in children.get(key, []):
            visit(kid, depth + 1)

    for root in roots:
        visit(root, 0)
    # 环内节点 / 悬空节点：没被遍历到的补在末尾，宁可多显示也不静默丢
    for key in nodes:
        visit(key, 0)
    return out


def _summarize(raw: Any) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw[:_MAX_SUMMARY_ROWS]:
        if not isinstance(item, dict):
            continue
        out.append({
            "operator": _text(item.get("operator")),
            "count": _int(item.get("count")),
            "rows": _int(item.get("rows")),
            "cost": _int(item.get("cost")),
        })
    return out


def build_plan_view(payload: Any) -> dict | None:
    """OCP explain 报文（或工具返回里的同构片段）→ 可视化视图；不成形返回 None。"""
    if not isinstance(payload, dict):
        return None
    raw = _pick(payload, "data", "plan_data")
    if not isinstance(raw, list) or not raw:
        return None
    nodes, children = _index_nodes(raw)
    if not nodes:
        return None
    roots = _roots(nodes, children, _pick(payload, "rootOperations", "root_operations"))
    ordered = _walk(nodes, children, roots)
    return {
        "uid": _text(payload.get("uid")),
        "sql_id": _text(_pick(payload, "sqlId", "sql_id")),
        "roots": roots,
        "nodes": [
            {
                "id": key,
                "depth": depth,
                "operator": _text(nodes[key].get("operator")),
                "name": _text(nodes[key].get("name")),
                "rows": _int(nodes[key].get("rows")),
                "cost": _int(nodes[key].get("cost")),
                "property": _text(nodes[key].get("property"), _MAX_PROPERTY_CHARS),
            }
            for key, depth in ordered[:_MAX_NODES]
        ],
        "summary": _summarize(_pick(payload, "planOperationSummery", "plan_operation_summery")),
        "node_count": len(ordered),
        "truncated": len(ordered) > _MAX_NODES,
    }


def extract_plan_view(name: str, output: Any) -> dict | None:
    """从工具返回里取计划视图；工具名不对、报文不成形或调用失败时返回 None。"""
    if name != PLAN_TOOL:
        return None
    content = getattr(output, "content", output)
    if isinstance(content, list):  # 内容块列表（新版 langchain 可能这样返回）
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    if not isinstance(content, str):
        return None
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("ok") is not True:
        return None
    items = data.get("items")
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return None
    return build_plan_view(items[0])