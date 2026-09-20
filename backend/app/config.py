"""配置加载：YAML 文件 + .env + 环境变量覆盖。

优先级：环境变量 > YAML > 默认值。
空字符串环境变量不覆盖 YAML 中已有非空值。
env=None 时先加载 backend/.env（若存在）再读 os.environ。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _as_bool(v: object, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    raise ValueError(f"invalid boolean value {v!r}: expected one of {sorted(_BOOL_TRUE | _BOOL_FALSE)}")


@dataclass
class OcpConfig:
    provider: str = "mock"
    base_url: str = ""
    username: str = ""
    password: str = ""
    verify_ssl: bool = True


@dataclass
class SqlConfig:
    provider: str = "mock"
    host: str = ""
    port: int = 3306
    username: str = ""
    password: str = ""
    db_name: str = None
    connect_timeout: int = 5
    query_timeout_seconds: int = 10
    max_rows: int = 200


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    max_input_tokens: int = 128000  # 模型最大输入 token 数，用于触发上下文压缩

    @property
    def is_configured(self) -> bool:
        """base_url/api_key/model 三者均非空（去空白后）才算配置齐全。"""
        return all(bool(getattr(self, field).strip()) for field in ("base_url", "api_key", "model"))


@dataclass
class AgentConfig:
    send_row_data: bool = True
    max_seconds: int = 120
    # 数据库操作人工确认（HITL）：true 时 DB 工具执行前需前端批准
    confirm_db_ops: bool = True
    confirm_timeout_seconds: int = 120
    # langgraph 图最大递归步数（防失控循环）：过低会误触顶（旧版 langgraph 默认 25），过高失去保护
    recursion_limit: int = 100


@dataclass
class Settings:
    ocp: OcpConfig = field(default_factory=OcpConfig)
    sql_ro: SqlConfig = field(default_factory=SqlConfig)
    meta_db: SqlConfig = field(default_factory=SqlConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config.yaml"


def _default_dotenv_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"


def _env_nonempty(env: Mapping[str, str], key: str) -> str | None:
    v = env.get(key)
    return v if v is not None and v.strip() != "" else None


def _maybe_load_dotenv(dotenv_path: Path | None) -> None:
    p = dotenv_path or _default_dotenv_path()
    if not p.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(p, override=False)


def load_settings(
    config_path: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    dotenv_path: str | None = None,
) -> Settings:
    if env is None:
        _maybe_load_dotenv(Path(dotenv_path) if dotenv_path else None)
        env = dict(os.environ)
    else:
        env = dict(env)

    path = Path(config_path) if config_path else _default_config_path()
    data: dict = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    ocp_y = data.get("ocp", {}) or {}
    sql_ro_y = data.get("sql_ro", {}) or {}
    meta_db_y = data.get("meta_db", {}) or {}
    llm_y = data.get("llm", {}) or {}
    agent_y = data.get("agent", {}) or {}

    verify_ssl_env = _env_nonempty(env, "OCP_VERIFY_SSL")
    send_row_data_env = _env_nonempty(env, "SEND_ROW_DATA")

    return Settings(
        ocp=OcpConfig(
            provider=_env_nonempty(env, "OCP_PROVIDER") or ocp_y.get("provider", "real"),
            base_url=_env_nonempty(env, "OCP_BASE_URL") or ocp_y.get("base_url", ""),
            username=_env_nonempty(env, "OCP_USERNAME") or ocp_y.get("username", ""),
            password=_env_nonempty(env, "OCP_PASSWORD") or ocp_y.get("password", ""),
            verify_ssl=_as_bool(
                verify_ssl_env if verify_ssl_env is not None else ocp_y.get("verify_ssl"),
                True,
            ),
        ),
        sql_ro=SqlConfig(
            provider=_env_nonempty(env, "SQL_RO_PROVIDER") or sql_ro_y.get("provider", "mock"),
            host=_env_nonempty(env, "SQL_RO_HOST_MAP") or sql_ro_y.get("host_map", ""),
            username=_env_nonempty(env, "SQL_RO_USERNAME") or sql_ro_y.get("username", ""),
            password=_env_nonempty(env, "SQL_RO_PASSWORD") or sql_ro_y.get("password", ""),
            connect_timeout=int(sql_ro_y.get("connect_timeout", 5)),
            query_timeout_seconds=int(sql_ro_y.get("query_timeout_seconds", 10)),
        ),
        meta_db=SqlConfig(
            provider=_env_nonempty(env, "META_DB_PROVIDER") or meta_db_y.get("provider", "mock"),
            host=_env_nonempty(env, "META_DB_HOST") or meta_db_y.get("host", ""),
            port=_env_nonempty(env, "META_DB_PORT") or meta_db_y.get("port", 3306),
            username=_env_nonempty(env, "META_DB_USERNAME") or meta_db_y.get("username", ""),
            password=_env_nonempty(env, "META_DB_PASSWORD") or meta_db_y.get("password", ""),
            db_name=_env_nonempty(env, "META_DB_NAME") or meta_db_y.get("db_name", ""),
            connect_timeout=int(meta_db_y.get("connect_timeout", 5)),
            query_timeout_seconds=int(meta_db_y.get("query_timeout_seconds", 10)),
        ),
        llm=LLMConfig(
            base_url=_env_nonempty(env, "LLM_BASE_URL") or llm_y.get("base_url", ""),
            api_key=_env_nonempty(env, "LLM_API_KEY") or llm_y.get("api_key", ""),
            model=_env_nonempty(env, "LLM_MODEL") or llm_y.get("model", ""),
            temperature=float(llm_y.get("temperature", 0.0)),
            max_input_tokens=int(_env_nonempty(env, "LLM_MAX_INPUT_TOKENS") or llm_y.get("max_input_tokens", 128000)),
        ),
        agent=AgentConfig(
            send_row_data=_as_bool(
                send_row_data_env if send_row_data_env is not None else agent_y.get("send_row_data"),
                True,
            ),
            max_seconds=int(agent_y.get("max_seconds", 120)),
            confirm_db_ops=_as_bool(
                _env_nonempty(env, "CONFIRM_DB_OPS")
                if _env_nonempty(env, "CONFIRM_DB_OPS") is not None
                else agent_y.get("confirm_db_ops"),
                True,
            ),
            confirm_timeout_seconds=int(
                _env_nonempty(env, "CONFIRM_TIMEOUT_SECONDS") or agent_y.get("confirm_timeout_seconds", 120)
            ),
            recursion_limit=int(
                _env_nonempty(env, "RECURSION_LIMIT") or agent_y.get("recursion_limit", 100)
            ),
        ),
    )