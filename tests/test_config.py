from pathlib import Path

import pytest

from app.config import load_settings


def _write_yaml(tmp_path: Path, text: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_defaults_are_mock_when_no_file(tmp_path):
    s = load_settings(config_path=None, env={"OCP_PROVIDER": "mock"})
    assert s.ocp.provider == "mock"
    assert s.sql_ro.provider == "mock"
    assert s.llm.base_url == ""
    assert s.agent.send_row_data is True
    assert s.agent.max_seconds == 120


def test_yaml_is_loaded(tmp_path):
    path = _write_yaml(
        tmp_path,
        "sql_ro:\n  provider: real\n  max_rows: 50\nagent:\n  send_row_data: false\n",
    )
    s = load_settings(config_path=path, env={})
    assert s.sql_ro.provider == "real"
    assert s.sql_ro.max_rows == 50
    assert s.agent.send_row_data is False


def test_env_overrides_yaml(tmp_path):
    path = _write_yaml(tmp_path, "llm:\n  base_url: http://a\n  model: m1\n")
    s = load_settings(
        config_path=path,
        env={"LLM_MODEL": "m2", "LLM_API_KEY": "k", "OCP_PROVIDER": "real"},
    )
    assert s.llm.model == "m2"
    assert s.llm.api_key == "k"
    assert s.ocp.provider == "real"
    assert s.llm.base_url == "http://a"  # 未被环境变量覆盖时保留 yaml 值


def test_env_empty_string_does_not_clobber_yaml(tmp_path):
    path = _write_yaml(tmp_path, "llm:\n  api_key: from_yaml\n")
    s = load_settings(config_path=path, env={"LLM_API_KEY": ""})
    assert s.llm.api_key == "from_yaml"


def test_env_empty_string_does_not_clobber_yaml_bool(tmp_path):
    path = _write_yaml(
        tmp_path,
        "ocp:\n  verify_ssl: false\nagent:\n  send_row_data: false\n",
    )
    s = load_settings(config_path=path, env={"OCP_VERIFY_SSL": "", "SEND_ROW_DATA": ""})
    assert s.ocp.verify_ssl is False
    assert s.agent.send_row_data is False


def test_env_string_bool_overrides_yaml_true(tmp_path):
    # yaml explicitly true, env "false" must win (string path through _as_bool)
    path = _write_yaml(tmp_path, "agent:\n  send_row_data: true\n")
    s = load_settings(config_path=path, env={"SEND_ROW_DATA": "false"})
    assert s.agent.send_row_data is False


def test_invalid_bool_env_raises(tmp_path):
    path = _write_yaml(tmp_path, "ocp:\n  verify_ssl: true\n")
    with pytest.raises(ValueError):
        load_settings(config_path=path, env={"OCP_VERIFY_SSL": "flase"})


def test_dotenv_is_loaded_when_env_not_given(tmp_path, monkeypatch):
    envp = tmp_path / ".env"
    envp.write_text("LLM_MODEL=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    s = load_settings(config_path=None, env=None, dotenv_path=str(envp))
    assert s.llm.model == "from_dotenv"
    monkeypatch.delenv("LLM_MODEL", raising=False)


def test_llm_is_configured_strips_whitespace():
    from app.config import LLMConfig

    assert LLMConfig(base_url="x", api_key="k", model="m").is_configured is True
    assert LLMConfig(base_url="  ", api_key="k", model="m").is_configured is False
    assert LLMConfig().is_configured is False
