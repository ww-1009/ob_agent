"""执行计划差异（plan_diff）的单元测试。

夹具刻意用「代价是累计值」的真实语义：父节点 cost = 自身代价 + 所有子节点 cost。
这是本模块的核心难点 —— 只看根节点代价只能知道「整个计划变贵了」，要靠自身代价
（cost − 子节点 cost 之和）才能指到真正变贵的那个算子。
"""
from __future__ import annotations

import json

import pytest

from app.agent.plan_diff import diff_plan_views, view_to_tree


def _node(node_id, depth, operator, *, name="", rows=None, cost=None, prop=""):
    return {
        "id": node_id,
        "depth": depth,
        "operator": operator,
        "name": name,
        "rows": rows,
        "cost": cost,
        "property": prop,
    }


def _view(nodes, uid=""):
    return {
        "uid": uid,
        "sql_id": "",
        "roots": [nodes[0]["id"]] if nodes else [],
        "nodes": nodes,
        "summary": [],
        "node_count": len(nodes),
        "truncated": False,
    }


_INDEXED = "table_rows:10000, physical_range_rows:10, index_back_rows:12, avaiable_index_name[IDX_ORDERS]"
_FULL_SCAN = "table_rows:10000, physical_range_rows:10000, index_back_rows:0, output_rows:10000"

# 根 → 连接 → 两张表；T2 在 after 里丢索引退化成全表扫描，代价沿祖先传播
_BEFORE = _view(
    [
        _node(0, 0, "PHY_SCALAR_AGGREGATE", rows=1, cost=100),
        _node(1, 1, "PHY_HASH_JOIN", rows=1, cost=100),
        _node(2, 2, "PHY_TABLE_SCAN", name="T1", rows=10, cost=60, prop=_INDEXED),
        _node(3, 2, "PHY_TABLE_SCAN", name="T2", rows=10, cost=40, prop=_INDEXED),
    ],
    uid="uid-before",
)
_AFTER = _view(
    [
        _node(0, 0, "PHY_SCALAR_AGGREGATE", rows=1, cost=960),
        _node(1, 1, "PHY_HASH_JOIN", rows=1, cost=960),
        _node(2, 2, "PHY_TABLE_SCAN", name="T1", rows=10, cost=60, prop=_INDEXED),
        _node(3, 2, "PHY_TABLE_SCAN", name="T2", rows=10000, cost=900, prop=_FULL_SCAN),
    ],
    uid="uid-after",
)


def test_view_to_tree_rebuilds_nesting_from_depth():
    tree = view_to_tree(
        _view([_node(0, 0, "A"), _node(1, 1, "B"), _node(2, 2, "C"), _node(3, 1, "D")])
    )
    assert [node["operator"] for node in tree] == ["A"]
    assert [node["operator"] for node in tree[0]["children"]] == ["B", "D"]
    assert [node["operator"] for node in tree[0]["children"][0]["children"]] == ["C"]


def test_view_to_tree_tolerates_bad_input_and_missing_depth():
    assert view_to_tree(None) == []
    assert view_to_tree({"nodes": "nope"}) == []
    tree = view_to_tree({"nodes": [{"operator": "A"}, {"operator": "B"}]})
    # depth 缺失时按「当前栈顶的子节点」兜底，而不是丢掉节点
    assert [node["operator"] for node in tree] == ["A"]
    assert [node["operator"] for node in tree[0]["children"]] == ["B"]


def test_identical_plans_report_unchanged():
    diff = diff_plan_views(_BEFORE, _BEFORE)
    assert diff["verdict"] == "unchanged" and diff["changed"] is False
    assert diff["counts"] == {"changed": 0, "added": 0, "removed": 0, "same": 4}
    assert diff["regressions"] == [] and diff["access_changes"] == []
    assert any("未发现回归" in line for line in diff["highlights"])


def test_regression_points_at_the_leaf_that_lost_its_index():
    diff = diff_plan_views(_BEFORE, _AFTER)
    assert diff["verdict"] == "regressed" and diff["cost_ratio"] == pytest.approx(9.6)
    assert (diff["before"]["cost"], diff["after"]["cost"]) == (100, 960)
    # 只有真正变贵的那个叶子算子算「变更」，祖先的代价是被传播上去的，不该一起报
    assert diff["counts"] == {"changed": 1, "added": 0, "removed": 0, "same": 3}
    top = diff["regressions"][0]
    assert top["operator"] == "PHY_TABLE_SCAN" and top["name"] == "T2"
    assert (top["self_cost_before"], top["self_cost_after"]) == (40, 900)
    notes = " ".join(top["notes"])
    assert "可用索引" in notes and "已退化为全表扫描" in notes and "扫描行数 10 → 10000" in notes
    assert any("回归" in line for line in diff["highlights"])
    assert len(diff["access_changes"]) == 1


def test_reversed_comparison_reports_improvement():
    diff = diff_plan_views(_AFTER, _BEFORE)
    assert diff["verdict"] == "improved"
    assert diff["improved"][0]["self_cost_after"] < diff["improved"][0]["self_cost_before"]
    assert diff["regressions"] == []


def test_pruned_tree_keeps_only_the_changed_branch():
    diff = diff_plan_views(_BEFORE, _AFTER)
    root = diff["node_tree"][0]
    assert root["operator"] == "PHY_SCALAR_AGGREGATE" and root["status"] == "same"
    # 与变更无关的 T1 分支被裁掉，只留「根 → 连接 → T2」这条路
    join = root["children"][0]
    assert join["operator"] == "PHY_HASH_JOIN"
    assert [node["name"] for node in join["children"]] == ["T2"]
    assert join["children"][0]["status"] == "changed"
    assert join["children"][0]["pruned"] is True


def test_added_and_removed_operators_are_aligned_by_lcs():
    before = _view(
        [
            _node(0, 0, "PHY_ROOT", cost=50),
            _node(1, 1, "PHY_TABLE_SCAN", name="OLD", cost=50),
        ]
    )
    after = _view(
        [
            _node(0, 0, "PHY_ROOT", cost=30),
            _node(1, 1, "PHY_TABLE_SCAN", name="NEW", cost=30),
        ]
    )
    diff = diff_plan_views(before, after)
    assert diff["counts"] == {"changed": 0, "added": 1, "removed": 1, "same": 1}
    assert diff["removed"][0]["name"] == "OLD" and diff["added"][0]["name"] == "NEW"
    assert diff["verdict"] == "improved"  # 代价 50 → 30，但结构变了；verdict 只看代价方向
    assert any("新增算子" in line for line in diff["highlights"])


def test_missing_metrics_do_not_crash():
    before = _view([_node(0, 0, "PHY_ROOT")])
    after = _view([_node(0, 0, "PHY_ROOT")])
    diff = diff_plan_views(before, after)
    assert diff["verdict"] == "unchanged" and diff["cost_ratio"] is None
    assert diff["before"]["cost"] is None and diff["before"]["rows"] is None


def test_empty_plan_view_is_rejected():
    with pytest.raises(ValueError):
        diff_plan_views(_view([]), _AFTER)


def test_real_fixtures_expose_a_full_scan_regression():
    """真实 OCP 报文夹具（data/ocp_sql_explain*.json）也要能跑出同样结论。"""
    from pathlib import Path

    from app.agent.plan_view import build_plan_view

    data_dir = Path(__file__).resolve().parent.parent / "backend" / "data"
    before = build_plan_view(json.loads((data_dir / "ocp_sql_explain.json").read_text("utf-8")))
    after = build_plan_view(
        json.loads((data_dir / "ocp_sql_explain_after.json").read_text("utf-8"))
    )
    assert before is not None and after is not None
    before["uid"], after["uid"] = "uid-before", "uid-after"
    diff = diff_plan_views(before, after)
    assert diff["verdict"] == "regressed"
    assert diff["counts"] == {"changed": 1, "added": 0, "removed": 0, "same": 9}
    assert diff["regressions"][0]["name"] == "WRT(WARN_RULE_TOTAL_INDEX_N1)"
    assert diff["before"]["uid"] == "uid-before" and diff["after"]["uid"] == "uid-after"
    assert any("971070" in line for line in diff["highlights"])