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
    """真实 OCP 报文：树在 rootOperations（小写键），data 只是首字母大写的同序扁平节点表。"""
    view = build_plan_view(_fixture())
    assert view["node_count"] == 10
    assert view["truncated"] is False
    assert view["roots"] == ["0"]  # 只有 PHY_SCALAR_AGGREGATE 是根
    # 先序 + depth：两个 PHY_TABLE_SCAN 在 PHY_SORT 下，第三个在 PHY_GRANULE_ITERATOR 下
    assert [(n["operator"], n["depth"]) for n in view["nodes"]] == [
        ("PHY_SCALAR_AGGREGATE", 0),
        ("PHY_HASH_JOIN", 1),
        ("PHY_MERGE_JOIN", 2),
        ("PHY_SORT", 3),
        ("PHY_TABLE_SCAN", 4),  # ER（PHY_SORT 下）
        ("PHY_TABLE_SCAN", 3),  # SUQ（PHY_MERGE_JOIN 下）
        ("PHY_PX_FIFO_COORD", 2),
        ("PHY_PX_REDUCE_TRANSMIT", 3),
        ("PHY_GRANULE_ITERATOR", 4),
        ("PHY_TABLE_SCAN", 5),  # WRT(WARN_RULE_TOTAL_INDEX_N1)
    ]
    # 算子名不吃 data 里的缩进空格；ObjectName / Property 的字面量 "NULL" 归一成空
    assert view["nodes"][0]["name"] == "" and view["nodes"][0]["property"] == ""
    scan = view["nodes"][4]
    assert scan["name"] == "ER" and scan["rows"] == 5 and scan["cost"] == 6
    assert scan["property"].startswith("table_rows:37")
    assert view["nodes"][9]["name"] == "WRT(WARN_RULE_TOTAL_INDEX_N1)"
    # 真实报文不带 sqlId（那是 top_plan 的字段），uid 由工具参数补
    assert view["sql_id"] == ""
    # 汇总：报文只给算子名，且是「一行一个节点」的算子清单（10 行与 10 个节点同序），
    # 因此逐行用对应节点的 rows/cost 补齐；同一个算子出现多次就逐行各报自己的数字。
    assert view["summary"][0] == {
        "operator": "PHY_SCALAR_AGGREGATE", "count": 1, "rows": 1, "cost": 1958,
    }
    assert [s for s in view["summary"] if s["operator"] == "PHY_TABLE_SCAN"] == [
        {"operator": "PHY_TABLE_SCAN", "count": 1, "rows": 5, "cost": 6},  # ER
        {"operator": "PHY_TABLE_SCAN", "count": 1, "rows": 27, "cost": 4},  # SUQ
        {"operator": "PHY_TABLE_SCAN", "count": 1, "rows": 1, "cost": 1942},  # WRT(...)
    ]


def test_summary_falls_back_to_operator_aggregate():
    """汇总表若与节点表对不上（旧 mock 是「一行一个算子」），则按算子聚合补齐数字。"""
    view = build_plan_view({
        "data": [
            {"id": 1, "operator": "EXCHANGE OUT"},
            {"id": 2, "operator": "TABLE SCAN", "rows": 10, "cost": 5},
            {"id": 3, "operator": "TABLE SCAN", "rows": 4, "cost": 3},
        ],
        "rootOperations": [1],
        "planOperationSummery": [{"operator": "TABLE SCAN"}],
    })
    assert view["summary"] == [
        {"operator": "TABLE SCAN", "count": 2, "rows": 14, "cost": 8},
    ]


def test_roots_fall_back_to_nodes_without_parent():
    view = build_plan_view({
        "data": [
            {"id": 1, "operator": "EXCHANGE OUT", "children": [0]},
            {"id": 0, "operator": "TABLE SCAN"},
        ],
    })
    assert view["roots"] == ["1"]  # 节点 1 没有被任何人当子节点
    assert [n["operator"] for n in view["nodes"]] == ["EXCHANGE OUT", "TABLE SCAN"]
    assert [n["depth"] for n in view["nodes"]] == [0, 1]


def test_capitalized_flat_rows_are_normalised():
    """树缺失时退回 data：首字母大写的键也要能读出算子/行数/代价，而不是整列空白。"""
    payload = _fixture()
    payload.pop("rootOperations")
    view = build_plan_view(payload)
    assert view["node_count"] == 10
    assert view["nodes"][0]["operator"] == "PHY_SCALAR_AGGREGATE"
    assert (view["nodes"][0]["rows"], view["nodes"][0]["cost"]) == (1, 1958)
    assert view["nodes"][4]["name"] == "ER"


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
    assert view["node_count"] == 10 and view["roots"] == ["0"]


def test_extract_accepts_content_blocks():
    class _Out:
        content = [{"type": "text", "text": _tool_output(_fixture())}]

    assert extract_plan_view(PLAN_TOOL, _Out())["node_count"] == 10


def test_extract_skips_other_tools_and_failures():
    out = _tool_output(_fixture())
    assert extract_plan_view("get_slow_sql", out) is None
    assert extract_plan_view(PLAN_TOOL, _tool_output(_fixture(), ok=False)) is None
    assert extract_plan_view(PLAN_TOOL, "不是 JSON") is None
    assert extract_plan_view(PLAN_TOOL, json.dumps({"ok": True, "items": []})) is None
    assert extract_plan_view(PLAN_TOOL, None) is None