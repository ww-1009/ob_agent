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


def test_system_prompt_names_all_tools():
    from app.agent.tools import build_tools
    from app.tools.ocp.mock import MockOcpClient
    from app.tools.sql.mock import MockSqlExecutor

    tools = build_tools(MockOcpClient(), MockSqlExecutor())
    p = system_prompt()
    for t in tools:
        assert t.name in p
