from app.agent.prompt import system_prompt


def test_system_prompt_key_elements():
    p = system_prompt()
    assert "OceanBase" in p
    # 提示词是英文，但规则 1 仍要求用中文作答
    assert "read-only" in p
    assert "answer in Chinese" in p
    assert "markdown" in p.lower()


def test_system_prompt_no_placeholder():
    p = system_prompt()
    assert "TBD" not in p
    assert "TODO" not in p


def test_system_prompt_points_at_doc_dir_and_all_views():
    """文档入口指向 ./doc/ob_wiki，并指明检索工具（文件工具根目录是 ./doc）；
    Oracle 元数据必须查 ALL_* + owner，不能再用只覆盖自身对象的 USER_*。"""
    p = system_prompt()
    assert "./doc/ob_wiki" in p
    assert "search_docs" in p
    assert "read_doc" in p
    assert "index.md" in p
    # 导航页/README 是降权而不是禁用：清单类提问要用 include_index 打开
    assert "README.md" in p
    assert "include_index" in p
    assert "ALL_TAB_COLUMNS" in p
    assert "USER_TAB_COLUMNS" not in p
    assert "owner" in p


def test_system_prompt_names_key_db_tools():
    # prompt 只在规则里点名关键工具（不再逐条枚举 build_tools 的全部工具名）
    p = system_prompt()
    for name in ("execute_sql", "get_table_ddl", "get_tenant_info",
                 "get_cluster_resource_stats", "get_server_resource_stats"):
        assert name in p


def test_system_prompt_tells_model_to_compare_plans():
    """同一 SQL 变慢时要比计划，而不是肉眼比对/直接倒树。"""
    p = system_prompt()
    assert "get_sql_top_plan" in p
    assert "compare_plans" in p
    assert "verdict" in p
    assert "cost ratio" in p
    assert "_before / *_after" in p
