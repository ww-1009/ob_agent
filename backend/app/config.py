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
    # ---- 以下两项仅 Oracle 模式租户使用；MySQL 模式下被忽略 ----
    # driver: OCI 驱动（oracledb | cx_oracle）。默认 oracledb：瘦模式免客户端库，
    #         且 cx_Oracle 无 Python 3.11+ 轮子（详见 tools/sql/oracle.py）。
    driver: str = "oracledb"
    # service_name: Oracle 模式由 DSN 的 service name 指定租户；留空则用租户名。
    #         需要「租户#集群」或 DBA 自定义的 service name 时在此显式指定。
    service_name: str = ""


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
    # 等待人工确认的超时；必须严格小于 max_seconds（load_settings 会校验），
    # 否则整轮的 asyncio.timeout 总是先到，审批超时永远不会触发
    confirm_timeout_seconds: int = 90
    # langgraph 图最大递归步数（防失控循环）：过低会误触顶（旧版 langgraph 默认 25），过高失去保护
    recursion_limit: int = 100


@dataclass
class MemoryConfig:
    """会话记忆持久化（PostgreSQL / LangGraph 检查点）。

    enabled 为 false 或连接失败时，后端退回无状态模式（前端回传全量历史）。
    """

    enabled: bool = False
    # dsn 非空则优先；否则由 host/port/user/password/dbname 拼装
    dsn: str = ""
    host: str = "127.0.0.1"
    port: int = 5432
    user: str = ""
    password: str = ""
    dbname: str = ""
    pool_min_size: int = 1
    pool_max_size: int = 5
    list_limit: int = 50       # GET /api/threads 默认上限
    messages_limit: int = 500  # GET /api/threads/{id}/messages 默认上限
    # >0 时在启动阶段清理更早的 audit_event 行；0 = 永久保留（默认，保持既有行为）
    audit_retention_days: int = 0
    # 启动打开连接的等待上限与尝试次数：失败要尽快降级，不能让启动被 PG 拖住
    # （最坏耗时 ≈ open_attempts × open_timeout_seconds + 退避）
    open_timeout_seconds: int = 5
    open_attempts: int = 2

    def build_dsn(self) -> str:
        """拼装连接串；分项拼装交给 psycopg 处理转义（避免手写 URL 编码出错）。"""
        if self.dsn.strip():
            return self.dsn.strip()
        if not self.dbname.strip():
            raise ValueError("memory 未配置完整：缺少 dbname（或直接给 memory.dsn）")
        # 延迟导入：未安装 psycopg 时仍可 import 本模块（记忆关闭场景不硬依赖）
        from psycopg.conninfo import make_conninfo

        return make_conninfo(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
        )


@dataclass
class AuthConfig:
    """访问控制（最小方案）：静态 Bearer 令牌保护除 /api/health 外的所有 /api 路由。

    enabled=true 而 token 为空时启动直接失败（fail closed），避免「以为开了其实没开」。
    """

    enabled: bool = False
    token: str = ""


@dataclass
class Settings:
    ocp: OcpConfig = field(default_factory=OcpConfig)
    sql_ro: SqlConfig = field(default_factory=SqlConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)


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


def _validate_agent(agent: AgentConfig) -> None:
    """校验 agent 配置的自洽性（fail fast）。

    审批超时必须严格短于整轮上限：两者相等时（旧默认值都是 120）审批超时永远不会
    先触发，先到的是整轮的 asyncio.timeout，用户看到的是「执行超时」而不是「审批
    超时」，人工确认形同虚设，审计里也看不出用户到底点没点。
    """
    if not agent.confirm_db_ops:
        return
    if agent.confirm_timeout_seconds <= 0:
        raise ValueError("agent.confirm_timeout_seconds 必须大于 0")
    if agent.confirm_timeout_seconds >= agent.max_seconds:
        raise ValueError(
            "agent.confirm_timeout_seconds 必须小于 agent.max_seconds："
            f"当前 {agent.confirm_timeout_seconds} >= {agent.max_seconds}，"
            "否则审批超时不会先触发，人工确认会退化成整轮执行超时"
        )


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
    llm_y = data.get("llm", {}) or {}
    agent_y = data.get("agent", {}) or {}
    memory_y = data.get("memory", {}) or {}
    auth_y = data.get("auth", {}) or {}

    verify_ssl_env = _env_nonempty(env, "OCP_VERIFY_SSL")
    send_row_data_env = _env_nonempty(env, "SEND_ROW_DATA")
    memory_enabled_env = _env_nonempty(env, "MEMORY_ENABLED")

    settings = Settings(
        ocp=OcpConfig(
            # 缺省 mock：无 config.yaml/.env 时「开箱即用」按 README 的离线演示跑通夹具，
            # 与 OcpConfig.provider 默认值及 sql_ro 保持一致（此前回落 real 会导致
            # base_url 为空仍走 real，OCP 工具必报配置缺失）。
            provider=_env_nonempty(env, "OCP_PROVIDER") or ocp_y.get("provider", "mock"),
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
            # max_rows 此前只在两个 README 与 YAML 里出现，代码从未读取 → 改 YAML 无效
            max_rows=int(sql_ro_y.get("max_rows", 200)),
            driver=_env_nonempty(env, "SQL_RO_DRIVER") or sql_ro_y.get("driver", "oracledb"),
            service_name=_env_nonempty(env, "SQL_RO_SERVICE_NAME") or sql_ro_y.get("service_name", ""),
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
                _env_nonempty(env, "CONFIRM_TIMEOUT_SECONDS") or agent_y.get("confirm_timeout_seconds", 90)
            ),
            recursion_limit=int(
                _env_nonempty(env, "RECURSION_LIMIT") or agent_y.get("recursion_limit", 100)
            ),
        ),
        memory=MemoryConfig(
            enabled=_as_bool(
                memory_enabled_env if memory_enabled_env is not None else memory_y.get("enabled"),
                False,
            ),
            dsn=_env_nonempty(env, "MEMORY_DSN") or memory_y.get("dsn", ""),
            host=_env_nonempty(env, "MEMORY_HOST") or memory_y.get("host", "127.0.0.1"),
            port=int(_env_nonempty(env, "MEMORY_PORT") or memory_y.get("port", 5432)),
            user=_env_nonempty(env, "MEMORY_USER") or memory_y.get("user", ""),
            password=_env_nonempty(env, "MEMORY_PASSWORD") or memory_y.get("password", ""),
            dbname=_env_nonempty(env, "MEMORY_DBNAME") or memory_y.get("dbname", ""),
            pool_min_size=int(memory_y.get("pool_min_size", 1)),
            pool_max_size=int(memory_y.get("pool_max_size", 5)),
            list_limit=int(memory_y.get("list_limit", 50)),
            messages_limit=int(memory_y.get("messages_limit", 500)),
            audit_retention_days=int(
                _env_nonempty(env, "MEMORY_AUDIT_RETENTION_DAYS")
                or memory_y.get("audit_retention_days", 0)
            ),
            open_timeout_seconds=int(
                _env_nonempty(env, "MEMORY_OPEN_TIMEOUT_SECONDS")
                or memory_y.get("open_timeout_seconds", 5)
            ),
            open_attempts=int(
                _env_nonempty(env, "MEMORY_OPEN_ATTEMPTS") or memory_y.get("open_attempts", 2)
            ),
        ),
        auth=AuthConfig(
            enabled=_as_bool(
                _env_nonempty(env, "AUTH_ENABLED")
                if _env_nonempty(env, "AUTH_ENABLED") is not None
                else auth_y.get("enabled"),
                False,
            ),
            token=_env_nonempty(env, "AUTH_TOKEN") or auth_y.get("token", ""),
        ),
    )
    _validate_agent(settings.agent)
    return settings