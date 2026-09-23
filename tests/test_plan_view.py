"""执行计划视图归一化（plan_view.py）测试。

主输入用 backend/data/ocp_sql_explain.json（mock 与真实 OCP 同构），另覆盖真实报文里
可能出现的变体：缺 rootOperations、内联子节点、环、id 缺失、超节点上限。
"""
import json
from pathlib import Path

from app.agent.plan_view import PLAN_TOOL, build_plan_view, extract_plan_view

_FIXTURE = Path(__file__).resolve().parents[1] / "backend" / "data" / "ocp_sql_explain.json"


def _fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _tool_output(payload: dict, *, ok: bool = True) -> str:
    return json.dumps({
        "ok": ok,
        "items": [{
            "plan_data": payload["data"],
            "plan_operation_summery": payload["planOperationSummery"],
            "root_operations": payload["rootOperations"],
        }],
    })


def test_build_view_from_fixture():
    view = build_plan_view(_fixture())
    assert view["node_count"] == 2
    assert view["truncated"] is False
    assert view["roots"] == ["1"]
    # 先序 + depth：EXCHANGE OUT 是根，TABLE SCAN 是它的子节点
    assert [(n["operator"], n["depth"]) for n in view["nodes"]] == [
        ("EXCHANGE OUT", 0),
        ("TABLE SCAN", 1),
    ]
    scan = view["nodes"][1]
    assert scan["name"] == "orders" and scan["rows"] == 1280 and scan["cost"] == 48210
    assert "is_index_back=false" in scan["property"]
    assert view["sql_id"] == "sq-scan-orders-1"
    assert view["summary"][0] == {"operator": "TABLE SCAN", "count": 1, "rows": 1280, "cost": 48210}
    assert view["summary"][1] == {"operator": "EXCHANGE OUT", "count": 1, "rows": 1280, "cost": 20}


def test_roots_fall_back_to_nodes_without_parent():
    payload = _fixture()
    payload.pop("rootOperations")
    view = build_plan_view(payload)
    assert view["roots"] == ["1"]  # 节点 1 没有被任何人当子节点
    assert [n["operator"] for n in view["nodes"]] == ["EXCHANGE OUT", "TABLE SCAN"]


def test_inline_child_objects_are_collected():
    view = build_plan_view({
        "data": [{
            "id": "p",
            "operator": "EXCHANGE OUT",
            "children": [{"id": "c", "operator": "TABLE SCAN", "rows": 3}],
        }],
    })
    assert [(n["id"], n["depth"]) for n in view["nodes"]] == [("p", 0), ("c", 1)]
    assert view["nodes"][1]["rows"] == 3


def test_cycle_terminates_and_keeps_every_node():
    view = build_plan_view({
        "data": [
            {"id": 1, "operator": "A", "children": [2]},
            {"id": 2, "operator": "B", "children": [1]},  # 环
        ],
        "rootOperations": [1],
    })
    assert [(n["id"], n["depth"]) for n in view["nodes"]] == [("1", 0), ("2", 1)]


def test_missing_id_falls_back_to_position():
    view = build_plan_view({"data": [{"operator": "A"}, {"operator": "B", "children": ["node-0"]}]})
    assert [n["id"] for n in view["nodes"]] == ["node-1", "node-0"]
    assert view["nodes"][1]["depth"] == 1


def test_string_numbers_are_coerced():
    view = build_plan_view({"data": [{"id": 1, "operator": "A", "rows": "1280", "cost": "48210.9"}]})
    assert view["nodes"][0]["rows"] == 1280 and view["nodes"][0]["cost"] == 48210


def test_node_cap_truncates_but_reports_full_count():
    data = [{"id": i, "operator": f"OP{i}", "children": [i + 1]} for i in range(250)]
    view = build_plan_view({"data": data, "rootOperations": [0]})
    assert view["node_count"] == 250
    assert len(view["nodes"]) == 200
    assert view["truncated"] is True


def test_long_property_is_capped():
    view = build_plan_view({"data": [{"id": 1, "operator": "A", "property": "x" * 500}]})
    assert len(view["nodes"][0]["property"]) == 241  # 240 + 省略号


def test_non_plan_payload_returns_none():
    for payload in (None, {}, {"data": []}, {"data": "oops"}, {"data": [None, 1]}, "not a dict"):
        assert build_plan_view(payload) is None


def test_extract_from_tool_output():
    view = extract_plan_view(PLAN_TOOL, _tool_output(_fixture()))
    assert view["node_count"] == 2 and view["roots"] == ["1"]


def test_extract_accepts_content_blocks():
    class _Out:
        content = [{"type": "text", "text": _tool_output(_fixture())}]

    assert extract_plan_view(PLAN_TOOL, _Out())["node_count"] == 2


def test_extract_skips_other_tools_and_failures():
    out = _tool_output(_fixture())
    assert extract_plan_view("get_slow_sql", out) is None
    assert extract_plan_view(PLAN_TOOL, _tool_output(_fixture(), ok=False)) is None
    assert extract_plan_view(PLAN_TOOL, "不是 JSON") is None
    assert extract_plan_view(PLAN_TOOL, json.dumps({"ok": True, "items": []})) is None
    assert extract_plan_view(PLAN_TOOL, None) is None