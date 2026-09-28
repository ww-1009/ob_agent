"""两份执行计划的差异与回归检测。

输入是 :func:`app.agent.plan_view.build_plan_view` 的输出（**先序 + depth 的扁平节点表**），
本模块负责：

1. 由 ``(depth, 顺序)`` 重建树（视图本身不带 children，重建规则与前端渲染一致）；
2. 同层按 ``(operator, objectName)`` 做 LCS 对齐，区分「变更 / 新增 / 移除」；
3. 汇总总代价、行数、算子数，并挑出两类最高信号：
   - **自身代价**（``cost`` 减去所有子节点 ``cost``）的增量 —— OCP 的 ``cost`` 是含子节点的
     累计值，直接比根节点代价只能看出「整个计划变贵了」，自身代价才能指到真正变贵的那个算子；
   - **索引/扫描方式变化** —— ``avaiable_index_name``（OCP 拼写如此）丢失、``physical_range_rows``
     涨到 ``table_rows`` 量级即退化为全表扫描、``unstable_index_name`` 出现等。

为什么不用「比较两份 JSON」让模型自己看：一份真实计划有几十个节点、每个节点一行 property，
两份一起下发既挤上下文又容易看漏；这里只把「变了的节点 + 结论」交给模型。
"""
from __future__ import annotations

import re
from typing import Any

# 计入摘要的条目上限（清单越长模型越容易只看开头）
_MAX_ITEMS = 10
# node_diff 节点上限：真实计划节点数已有上限（plan_view._MAX_NODES=200）
_MAX_DIFF_NODES = 200
# 「显著变化」阈值：行数/代价相对变化小于它、且绝对值也没动，就按「未变」处理，
# 避免代价传播（父节点跟着子节点一起涨）把每个祖先都标成 changed
_MATERIAL_RATIO = 0.2
# 代价比例达到它就判回归 / 改善
_VERDICT_RATIO = 1.2

_AVAIL_INDEX_RE = re.compile(r"avai[l]?able_index_name\s*\[([^\]]*)\]", re.I)
_UNSTABLE_INDEX_RE = re.compile(r"unstable_index_name\s*\[([^\]]*)\]", re.I)
_ACCESS_RE = re.compile(r"access\s*\(\s*([^)]*)\)", re.I)
_NUM_FIELD_RE = {
    name: re.compile(rf"\b{name}\s*:\s*(\d+)", re.I)
    for name in (
        "table_rows",
        "physical_range_rows",
        "logical_range_rows",
        "index_back_rows",
        "output_rows",
        "is_index_back",
        "use_das",
    )
}


def _int(value: Any) -> int | None:
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


def _ratio(before: int | None, after: int | None) -> float | None:
    """after / before；除零与缺失都给 None（调用方按「无法比较」处理）。"""
    if before is None or after is None:
        return None
    if before == 0:
        return None if after == 0 else float("inf")
    return after / before


def _material(before: int | None, after: int | None) -> bool:
    """相对变化是否够显著（绝对值至少动了 1）。"""
    if before is None or after is None:
        return before != after
    delta = after - before
    if delta == 0:
        return False
    if abs(delta) <= 1:
        return abs(delta) == 1
    ratio = _ratio(before, after)
    if ratio is None:
        return True
    return abs(ratio - 1.0) >= _MATERIAL_RATIO


def _index_list(text: str) -> list[str] | None:
    match = _AVAIL_INDEX_RE.search(text or "")
    if not match:
        return None
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def _unstable_list(text: str) -> list[str] | None:
    match = _UNSTABLE_INDEX_RE.search(text or "")
    if not match:
        return None
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def _prop_fields(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, pattern in _NUM_FIELD_RE.items():
        match = pattern.search(text or "")
        if match:
            out[name] = int(match.group(1))
    return out


def _label(node: dict) -> str:
    operator = (node.get("operator") or "").strip() or "?"
    name = (node.get("name") or "").strip()
    return f"{operator}({name})" if name else operator


def _path_label(ancestors: list[str], node: dict) -> str:
    parts = ancestors + [_label(node)]
    if len(parts) > 4:  # 深计划只留首尾，避免路径本身比结论还长
        parts = parts[:1] + ["…"] + parts[-2:]
    return " > ".join(parts)


def view_to_tree(view: Any) -> list[dict]:
    """先序 + depth 的节点表 → 嵌套树。depth 缺失或跳级时按「当前栈顶的子节点」兜底。"""
    if not isinstance(view, dict):
        return []
    nodes = view.get("nodes")
    if not isinstance(nodes, list):
        return []
    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []
    for raw in nodes:
        if not isinstance(raw, dict):
            continue
        node = {
            "id": raw.get("id"),
            "operator": (raw.get("operator") or "").strip(),
            "name": (raw.get("name") or "").strip(),
            "rows": _int(raw.get("rows")),
            "cost": _int(raw.get("cost")),
            "property": raw.get("property") or "",
            "children": [],
        }
        depth = _int(raw.get("depth"))
        if depth is None:
            depth = (stack[-1][0] + 1) if stack else 0
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            stack[-1][1]["children"].append(node)
        else:
            roots.append(node)
        stack.append((depth, node))
    return roots


def _self_costs(roots: list[dict]) -> None:
    """就地写入 ``self_cost`` = 本节点 cost − 子节点 cost 之和（OCP 的 cost 是累计值）。"""

    def walk(node: dict) -> int:
        child_total = 0
        for child in node["children"]:
            child_total += walk(child)
        cost = node.get("cost")
        node["self_cost"] = None if cost is None else max(cost - child_total, 0)
        return cost if cost is not None else child_total

    for root in roots:
        walk(root)


def _totals(roots: list[dict]) -> dict:
    """计划总量：根节点代价/行数（多根求和）+ 节点数。"""
    cost = sum(root["cost"] for root in roots if root.get("cost") is not None)
    rows = sum(root["rows"] for root in roots if root.get("rows") is not None)

    def count(node: dict) -> int:
        return 1 + sum(count(child) for child in node["children"])

    return {
        "node_count": sum(count(root) for root in roots),
        "cost": cost if any(root.get("cost") is not None for root in roots) else None,
        "rows": rows if any(root.get("rows") is not None for root in roots) else None,
    }


def _key(node: dict) -> tuple[str, str]:
    return (node.get("operator", "").upper(), node.get("name", "").upper())


def _lcs(a: list, b: list) -> list[tuple[int, int]]:
    """最长公共子序列的下标配对（计划节点很少，O(n·m) 足够）。"""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if _key(a[i]) == _key(b[j]):
                table[i][j] = table[i + 1][j + 1] + 1
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if _key(a[i]) == _key(b[j]):
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _property_notes(before: dict, after: dict) -> list[str]:
    """只报「看得出因果」的属性变化：索引可用性、扫描行数、回表、DAS。"""
    notes: list[str] = []
    old_prop, new_prop = before.get("property") or "", after.get("property") or ""

    old_idx, new_idx = _index_list(old_prop), _index_list(new_prop)
    if old_idx != new_idx and (old_idx or new_idx):
        def show(items: list[str] | None) -> str:
            if not items:
                return "无"
            return "[" + ", ".join(items[:4]) + ("…" if len(items) > 4 else "") + "]"

        notes.append(f"可用索引 {show(old_idx)} → {show(new_idx)}")
        if old_idx and not new_idx:
            notes.append("索引不可用，已退化为全表扫描")

    old_unstable, new_unstable = _unstable_list(old_prop), _unstable_list(new_prop)
    if old_unstable != new_unstable and (old_unstable or new_unstable):
        notes.append(
            "统计信息不稳定索引 "
            f"{'、'.join(old_unstable) or '无'} → {'、'.join(new_unstable) or '无'}"
        )

    old_fields, new_fields = _prop_fields(old_prop), _prop_fields(new_prop)
    for field, label in (
        ("physical_range_rows", "扫描行数"),
        ("index_back_rows", "回表行数"),
        ("output_rows", "输出行数"),
    ):
        old_value, new_value = old_fields.get(field), new_fields.get(field)
        if _material(old_value, new_value):
            notes.append(f"{label} {old_value} → {new_value}")

    old_access, new_access = _ACCESS_RE.search(old_prop), _ACCESS_RE.search(new_prop)
    if (old_access or new_access) and (
        (old_access.group(1).strip() if old_access else "")
        != (new_access.group(1).strip() if new_access else "")
    ):
        notes.append(
            f"access {'/'.join(filter(None, [old_access.group(1).strip() if old_access else ''])) or '无'}"
            f" → {'/'.join(filter(None, [new_access.group(1).strip() if new_access else ''])) or '无'}"
        )
    if old_fields.get("is_index_back") != new_fields.get("is_index_back"):
        notes.append(
            f"index_back {old_fields.get('is_index_back', '无')} → {new_fields.get('is_index_back', '无')}"
        )
    return notes


def _entry(status: str, path: str, before: dict | None, after: dict | None, notes: list[str]) -> dict:
    node = after or before or {}
    return {
        "status": status,
        "path": path,
        "operator": node.get("operator", ""),
        "name": node.get("name", ""),
        "rows_before": (before or {}).get("rows"),
        "rows_after": (after or {}).get("rows"),
        "cost_before": (before or {}).get("cost"),
        "cost_after": (after or {}).get("cost"),
        "self_cost_before": (before or {}).get("self_cost"),
        "self_cost_after": (after or {}).get("self_cost"),
        "notes": notes,
    }


def _walk_diff(
    before_children: list[dict],
    after_children: list[dict],
    ancestors: list[str],
    out: list[dict],
    changed_ids: set,
) -> bool:
    """同层 LCS 对齐后逐对比较；返回「本层是否有变更」（供剪枝）。"""
    changed = False
    pairs = dict(_lcs(before_children, after_children))
    consumed_before = set(pairs)
    consumed_after = set(pairs.values())
    for index, child in enumerate(before_children):
        if index not in consumed_before:
            out.append(_entry("removed", _path_label(ancestors, child), child, None, ["该算子在新计划中消失"]))
            changed = True
    for index, child in enumerate(after_children):
        if index not in consumed_after:
            out.append(_entry("added", _path_label(ancestors, child), None, child, ["新增算子"]))
            changed_ids.add(child.get("id"))
            changed = True

    for i in sorted(pairs):
        before_node, after_node = before_children[i], after_children[pairs[i]]
        path = _path_label(ancestors, after_node)
        notes: list[str] = []
        if before_node.get("operator") != after_node.get("operator"):
            notes.append(f"算子 {before_node.get('operator')} → {after_node.get('operator')}")
        if before_node.get("name") != after_node.get("name"):
            notes.append(f"对象 {before_node.get('name') or '无'} → {after_node.get('name') or '无'}")
        if _material(before_node.get("rows"), after_node.get("rows")):
            notes.append(f"行数估计 {before_node.get('rows')} → {after_node.get('rows')}")
        if _material(before_node.get("self_cost"), after_node.get("self_cost")):
            notes.append(f"自身代价 {before_node.get('self_cost')} → {after_node.get('self_cost')}")
        notes.extend(_property_notes(before_node, after_node))
        status = "changed" if notes else "same"
        out.append(_entry(status, path, before_node, after_node, notes))
        changed = status == "changed" or changed
        if status == "changed":
            changed_ids.add(after_node.get("id"))
        sub_changed = _walk_diff(
            before_node["children"],
            after_node["children"],
            ancestors + [_label(before_node)],
            out,
            changed_ids,
        )
        changed = changed or sub_changed
    return changed


def _prune_diff_tree(roots_after: list[dict], changed_ids: set) -> list[dict]:
    """前端用的精简树：只保留变更节点及其祖先路径，其余分支整段裁掉。"""
    def build(node: dict, ancestors: list[str]) -> dict | None:
        kids = [built for child in node["children"] if (built := build(child, ancestors + [_label(node)]))]
        is_changed = node.get("id") in changed_ids
        if not is_changed and not kids:
            # 与变更无关的分支整段裁掉：树只留「变更节点 + 它们的祖先路径」
            return None
        return {
            "operator": node.get("operator", ""),
            "name": node.get("name", ""),
            "rows": node.get("rows"),
            "cost": node.get("cost"),
            "self_cost": node.get("self_cost"),
            "status": "changed" if is_changed else "same",
            "pruned": not kids,
            "children": kids,
        }

    out = []
    for root in roots_after:
        built = build(root, [])
        if built is not None:
            out.append(built)
    return out


def diff_plan_views(before: Any, after: Any, *, limit: int = _MAX_ITEMS) -> dict:
    """两份计划视图 → 差异摘要（verdict / highlights / 回归清单 / 结构化 node_diff）。"""
    roots_before, roots_after = view_to_tree(before), view_to_tree(after)
    if not roots_before or not roots_after:
        raise ValueError("执行计划视图为空或不完整，无法对比")
    _self_costs(roots_before)
    _self_costs(roots_after)

    entries: list[dict] = []
    changed_ids: set = set()
    _walk_diff(roots_before, roots_after, [], entries, changed_ids)

    totals_before, totals_after = _totals(roots_before), _totals(roots_after)
    cost_ratio = _ratio(totals_before["cost"], totals_after["cost"])
    if not entries or all(entry["status"] == "same" for entry in entries):
        verdict = "unchanged"
    elif cost_ratio is None:
        verdict = "changed"
    elif cost_ratio >= _VERDICT_RATIO:
        verdict = "regressed"
    elif cost_ratio <= 1 / _VERDICT_RATIO:
        verdict = "improved"
    else:
        verdict = "changed"

    def delta(entry: dict) -> int:
        old = entry.get("self_cost_before")
        new = entry.get("self_cost_after")
        if old is None and new is None:
            return 0
        return (new or 0) - (old or 0)

    regressed = sorted((e for e in entries if delta(e) > 0), key=delta, reverse=True)
    improved = sorted((e for e in entries if delta(e) < 0), key=delta)
    changed = [e for e in entries if e["status"] == "changed"]
    added = [e for e in entries if e["status"] == "added"]
    removed = [e for e in entries if e["status"] == "removed"]

    highlights: list[str] = []
    if totals_before["cost"] is not None and totals_after["cost"] is not None:
        arrow = f"总代价 {totals_before['cost']} → {totals_after['cost']}"
        if cost_ratio not in (None, float("inf")):
            arrow += f"（{cost_ratio:.2f} 倍）"
        highlights.append(arrow)
    if totals_before["rows"] is not None and _material(totals_before["rows"], totals_after["rows"]):
        highlights.append(f"根节点行数估计 {totals_before['rows']} → {totals_after['rows']}")
    if totals_before["node_count"] != totals_after["node_count"]:
        highlights.append(f"算子数 {totals_before['node_count']} → {totals_after['node_count']}")
    for entry in regressed[:3]:
        detail = "；".join(entry["notes"][:3])
        highlights.append(f"回归：{entry['path']}" + (f" —— {detail}" if detail else ""))
    for entry in improved[:2]:
        detail = "；".join(entry["notes"][:3])
        highlights.append(f"改善：{entry['path']}" + (f" —— {detail}" if detail else ""))
    if added:
        highlights.append("新增算子：" + "、".join(e["path"] for e in added[:3]))
    if removed:
        highlights.append("移除算子：" + "、".join(e["path"] for e in removed[:3]))
    if verdict == "unchanged":
        highlights.append("两份计划的结构与指标一致，未发现回归")

    access_changes = [
        e for e in entries
        if any("索引" in note or "扫描" in note or "access" in note for note in e["notes"])
    ]
    node_tree = _prune_diff_tree(roots_after, changed_ids)
    return {
        "verdict": verdict,
        "changed": verdict != "unchanged",
        "before": {"uid": (before or {}).get("uid", ""), **totals_before},
        "after": {"uid": (after or {}).get("uid", ""), **totals_after},
        "cost_ratio": None if cost_ratio in (None, float("inf")) else round(cost_ratio, 3),
        "highlights": highlights,
        "regressions": regressed[:limit],
        "improved": improved[:limit],
        "changed_nodes": changed[:limit],
        "added": added[:limit],
        "removed": removed[:limit],
        "access_changes": access_changes[:limit],
        "counts": {
            "changed": len(changed),
            "added": len(added),
            "removed": len(removed),
            "same": sum(1 for e in entries if e["status"] == "same"),
        },
        "node_tree": node_tree[:_MAX_DIFF_NODES],
    }