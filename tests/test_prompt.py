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
    assert "ALL_TAB_COLUMNS" in p
    assert "USER_TAB_COLUMNS" not in p
    assert "owner" in p


def test_system_prompt_names_key_db_tools():
    # prompt 只在规则里点名关键工具（不再逐条枚举 build_tools 的全部工具名）
    p = system_prompt()
    for name in ("execute_sql", "get_table_ddl", "get_tenant_info",
                 "get_cluster_resource_stats", "get_server_resource_stats"):
        assert name in p
