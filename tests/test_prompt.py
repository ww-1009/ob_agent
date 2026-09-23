from app.agent.prompt import system_prompt


def test_system_prompt_key_elements():
    p = system_prompt()
    assert "OceanBase" in p
    assert "只读" in p
    assert "markdown" in p.lower()


def test_system_prompt_no_placeholder():
    p = system_prompt()
    assert "TBD" not in p
    assert "TODO" not in p


def test_system_prompt_names_key_db_tools():
    # prompt 只在规则里点名关键工具（不再逐条枚举 build_tools 的全部工具名）
    p = system_prompt()
    for name in ("execute_sql", "get_table_ddl", "get_tenant_info"):
        assert name in p
