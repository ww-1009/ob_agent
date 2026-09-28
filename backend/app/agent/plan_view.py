"""执行计划的可视化视图（服务端归一化，前端只负责渲染）。

OCP 的 explain 报文有两种形态，本模块都收：

1. 真实 OCP 报文：``data`` 是与树同序的**扁平**节点表（键首字母大写：``Id`` / ``Operator`` /
   ``Rows`` / ``Cost`` / ``Property`` / ``ObjectName``，没有 ``children``），真正的树在
   ``rootOperations`` —— 嵌套 dict，键为小写 ``id`` / ``operator`` / ``children``::

       {"data": [{"Id": 0, "Operator": "PHY_SCALAR_AGGREGATE", "Rows": 1, "Cost": 1958}],
        "planOperationSummery": [{"operator": "PHY_SCALAR_AGGREGATE", "objectName": "NULL"}],
        "rootOperations": [{"id": 0, "operator": "PHY_SCALAR_AGGREGATE",
                            "children": [{"id": 1, "operator": "PHY_HASH_JOIN"}]}]}

2. 旧 mock 与联调期变体：``data`` 自带小写键与 ``children``（子节点 id 或内联子节点对象），
   ``rootOperations`` 只给根节点 id::

       {"data": [{"id": 1, "operator": "EXCHANGE OUT", "name": "distributed",
                  "children": [0]}],
        "planOperationSummery": [{"operator": "TABLE SCAN", "count": 1, ...}],
        "rootOperations": [1]}

直接丢给前端，前端要自己重建树、还要防环，还得把两种键名各写一遍。这里一次性归一成
「先序 + depth」的扁平列表与算子汇总，前端只做渲染。

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


def _blank(value: Any) -> bool:
    """OCP 用字面量 ``"NULL"`` 表示「没有」，与空值一样归一成空。"""
    text = "" if value is None else str(value)
    return text.strip() == "" or text.strip().upper() == "NULL"


def _display(item: dict, *keys: str, limit: int = _MAX_TEXT_CHARS) -> str:
    """取首个「有内容」的显示字段。

    键名按序尝试：真实报文用首字母大写（``Operator`` / ``ObjectName``），旧 mock 用小写；
    字面量 ``"NULL"`` 视为空，前端不必显示一串 ``NULL``。缩进用的前导空格也一并去掉。
    """
    for key in keys:
        value = item.get(key)
        if not _blank(value):
            return _text(value, limit).strip()
    return ""


def _children_of(item: dict) -> Any:
    return _pick(item, "children", "Children")


def _tree_of(declared: Any) -> list[dict] | None:
    """真实 OCP 报文的树在 ``rootOperations``（嵌套 dict）；``data`` 只是同序的扁平节点表。

    只有确实带嵌套子节点时才按树解析，否则交回 ``data``（兼容「data + 根 id 列表」的形态）。
    """
    if not isinstance(declared, list):
        return None
    items = [item for item in declared if isinstance(item, dict)]
    if not items or not any(_children_of(item) for item in items):
        return None
    return items


def _index_nodes(raw: list) -> tuple[dict[str, dict], dict[str, list[str]], list[str]]:
    """登记所有节点，返回（节点表, 每个节点的子键列表, 顶层键列表）。

    节点 id 缺失时用位置序号兜底；children 里既可能是子节点 id（OCP 实际报文），
    也可能是内联的子节点对象（联调期见过的变体），两种都收。键名大小写不敏感。
    """
    nodes: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    top: list[str] = []

    def register(item: Any, fallback: str) -> str | None:
        if not isinstance(item, dict):
            return None
        raw_id = _pick(item, "id", "Id")
        key = str(raw_id) if raw_id is not None else fallback
        if key in nodes:  # 重复引用：第一次为准
            return key
        nodes[key] = item
        kids: list[str] = []
        for idx, child in enumerate(_children_of(item) or []):
            if isinstance(child, dict):
                sub = register(child, f"{key}.{idx}")
                if sub is not None:
                    kids.append(sub)
            else:
                kids.append(str(child))
        children[key] = kids
        return key

    for i, item in enumerate(raw):
        key = register(item, f"node-{i}")
        if key is not None:
            top.append(key)
    return nodes, children, top


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


def _summarize(raw: Any, nodes: dict[str, dict]) -> list[dict]:
    """算子汇总。

    真实报文的 ``planOperationSummery`` 只给算子名与对象名（没有次数/行数/代价），且形态是
    「一行一个节点」的算子清单（算子名用前导空格表缩进），与节点表逐行对应：此时逐行用对应
    节点的 rows / cost 补齐。若它对不上节点表（旧 mock 是「一行一个算子」的聚合表），则按算子
    汇总补齐：count = 该算子的节点数，rows / cost = 各节点求和。报文自己带了数字时以报文为准。
    """
    if not isinstance(raw, list):
        return []
    rows = [item for item in raw[:_MAX_SUMMARY_ROWS] if isinstance(item, dict)]
    node_list = list(nodes.values())

    per_node = len(rows) == len(node_list) and all(
        _display(row, "operator", "Operator") == _display(node, "operator", "Operator")
        for row, node in zip(rows, node_list)
    )
    totals: dict[str, dict[str, int]] = {}
    for item in node_list:
        operator = _display(item, "operator", "Operator")
        if not operator:
            continue
        agg = totals.setdefault(operator, {"count": 0, "rows": 0, "cost": 0})
        agg["count"] += 1
        agg["rows"] += _int(_pick(item, "rows", "Rows")) or 0
        agg["cost"] += _int(_pick(item, "cost", "Cost")) or 0

    out = []
    for idx, item in enumerate(rows):
        operator = _display(item, "operator", "Operator")
        count = _int(_pick(item, "count", "Count"))
        row_count = _int(_pick(item, "rows", "Rows"))
        cost = _int(_pick(item, "cost", "Cost"))
        if per_node:
            node = node_list[idx]
            count = 1 if count is None else count
            if row_count is None:
                row_count = _int(_pick(node, "rows", "Rows"))
            if cost is None:
                cost = _int(_pick(node, "cost", "Cost"))
        else:
            agg = totals.get(operator, {})
            count = count if count is not None else agg.get("count")
            row_count = row_count if row_count is not None else agg.get("rows")
            cost = cost if cost is not None else agg.get("cost")
        out.append({
            "operator": operator,
            "count": count,
            "rows": row_count,
            "cost": cost,
        })
    return out


def build_plan_view(payload: Any) -> dict | None:
    """OCP explain 报文（或工具返回里的同构片段）→ 可视化视图；不成形返回 None。"""
    if not isinstance(payload, dict):
        return None
    declared = _pick(payload, "rootOperations", "root_operations")
    tree = _tree_of(declared)
    if tree is not None:
        # 真实报文：树在 rootOperations，data 只是与树同序的扁平节点表
        nodes, children, roots = _index_nodes(tree)
    else:
        raw = _pick(payload, "data", "plan_data")
        if not isinstance(raw, list) or not raw:
            return None
        nodes, children, top = _index_nodes(raw)
        if not nodes:
            return None
        roots = _roots(nodes, children, declared) or top
    if not nodes:
        return None
    ordered = _walk(nodes, children, roots)
    return {
        "uid": _text(payload.get("uid")),
        "sql_id": _text(_pick(payload, "sqlId", "sql_id")),
        "roots": roots,
        "nodes": [
            {
                "id": key,
                "depth": depth,
                "operator": _display(nodes[key], "operator", "Operator"),
                "name": _display(nodes[key], "name", "objectName", "ObjectName"),
                "rows": _int(_pick(nodes[key], "rows", "Rows")),
                "cost": _int(_pick(nodes[key], "cost", "Cost")),
                "property": _display(
                    nodes[key], "property", "Property", limit=_MAX_PROPERTY_CHARS
                ),
            }
            for key, depth in ordered[:_MAX_NODES]
        ],
        "summary": _summarize(
            _pick(payload, "planOperationSummery", "plan_operation_summery"), nodes
        ),
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