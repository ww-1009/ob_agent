"""配置加载：YAML 文件 + .env + 环境变量覆盖。

优先级：环境变量 > YAML > 默认值。
空字符串环境变量不覆盖 YAML 中已有非空值。
env=None 时先加载 backend/.env（若存在）再读 os.environ。
"""
from __future__ import annotations

import logging
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
    # ---- 以下一项仅 Oracle 模式租户使用；MySQL 模式下被忽略 ----
    # driver: OCI 驱动（oracledb | cx_oracle）。默认 oracledb：瘦模式免客户端库，
    #         且 cx_Oracle 无 Python 3.11+ 轮子（详见 tools/sql/oracle.py）。
    # 注意：Oracle 模式没有单独的 service name 配置项 —— DSN 的 service name 直接取
    #         工具（execute_sql / get_table_ddl）传入的 db_name，见 oracle.py:resolve_config。
    driver: str = "oracledb"


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
class EmbeddingConfig:
    """稠密一路的外部 Embedding 服务（OpenAI 兼容 POST {base_url}/embeddings）。

    is_configured **有意不要求 api_key**（区别于 LLMConfig）：Ollama / vLLM 等本地
    服务免鉴权，若把空 key 判成「未配置」，本地开发会被错误地禁用稠密一路。
    key 为空时请求不发 Authorization 头，由服务端决定是否拒绝。
    """

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # 必须与模型实际输出维度一致；不一致由启动/建索引侧报错并提示 --rebuild-vectors
    dims: int = 1024
    # 单次 /embeddings 请求的条数上限。DashScope 兼容模式硬上限为 20
    # （超限返回 400 invalid_parameter_error: batch size is invalid, it should not be larger than 20），
    # 故保守默认 16；自建 vLLM/Ollama 可调大。
    batch_size: int = 16           # 建索引时的批量大小
    concurrency: int = 4           # 建索引时的并发请求数
    timeout_seconds: int = 30
    query_cache_size: int = 512    # 查询向量 LRU 容量（0 = 不缓存）

    @property
    def is_configured(self) -> bool:
        """base_url 与 model 均非空（去空白后）才算可用。"""
        return all(bool(getattr(self, f).strip()) for f in ("base_url", "model"))

    @property
    def endpoint_url(self) -> str:
        return self.base_url.rstrip("/") + "/embeddings"


@dataclass
class RerankConfig:
    """后置重排。默认协议 Jina / Cohere 兼容（POST {base_url}/rerank）。

    protocol 说明（实测：DashScope「compatible-mode/v1」并不提供 /rerank，
    阿里云百炼的重排只能走原生形态）：
      jina：{base_url}/rerank，请求 {"model","query","documents","top_n"}，
            响应 results[].index / results[].relevance_score
      dashscope：{host}/api/v1/services/rerank/text-rerank/text-rerank，
            请求 {"model","input":{"query","documents"},"parameters":{"top_n",...}}，
            响应 output.results[].index / output.results[].relevance_score

    mode 语义：
      auto（默认）：配置齐全即启用，缺配置静默等价 off —— 不阻断启动，由 health 暴露降级
      off：显式关闭
      api：显式要求启用；配置不齐直接启动失败（fail closed，避免「以为开了其实没开」）
    """

    mode: str = "auto"
    protocol: str = "jina"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    top_n: int = 30                # 送入 rerank 的候选数
    timeout_seconds: float = 3.0   # 超时即保留融合原序（不报错）
    max_passage_chars: int = 500   # 单条 passage 截断上限

    @property
    def is_configured(self) -> bool:
        """base_url 与 model 均非空（api_key 可空，理由同 EmbeddingConfig）。"""
        return all(bool(getattr(self, f).strip()) for f in ("base_url", "model"))

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and self.is_configured

    @property
    def endpoint_url(self) -> str:
        base = self.base_url.rstrip("/")
        if self.protocol != "dashscope":
            return base + "/rerank"
        # 原生形态挂在 host 根下：从任何 compatible-mode / v1 形态的 base_url 反推 host
        if "/api/v1/services/rerank" in base:
            return base
        host = base.split("/compatible-mode")[0].rstrip("/")
        return host + "/api/v1/services/rerank/text-rerank/text-rerank"


@dataclass
class RetrievalConfig:
    """检索后端（Milvus Lite 单引擎：内建 BM25 稀疏 + FLOAT_VECTOR 稠密）。

    milvus_path 相对 backend/ 解析（锚点同 _default_config_path），不随启动 CWD 漂移。
    default_retriever 决定默认走哪一路：M7 删掉 FTS5 后是 "sparse"（唯一实测过门禁的一路）；
    dense/hybrid 的真实向量基线跑通后再改这里。
    """

    milvus_path: str = "doc/ob_wiki.milvus.db"   # milvus-lite 只认 .db 结尾的本地路径
    collection: str = "ob_chunks"
    meta_collection: str = "ob_meta"
    analyzer: str = "jieba"        # milvus-lite 3.2.1 的中文分析器类型名必须是 jieba
    dense_index_type: str = "IVF_FLAT"
    nlist: int = 128
    metric: str = "COSINE"
    pool_k: int = 50               # 每路融合前的候选数
    rrf_k: int = 60                # 自研 RRF 的平滑常数
    weight_sparse: float = 1.0
    weight_dense: float = 1.0
    query_timeout_seconds: int = 5
    query_cache_size: int = 512
    max_text_bytes: int = 8000     # 倒排文本上限；Milvus 的 VARCHAR max_length 实测按字符计，这里仍按字节保守守卫
    fingerprint_ttl_seconds: float = 5.0   # 「语料是否已变」查询期检查的缓存 TTL（秒）；0 = 每次查询重扫
    version_match_bonus: float = 8.0       # 取回后调整（进入 RRF 前）
    nav_section_penalty: float = 12.0
    nav_file_penalty: float = 40.0
    default_retriever: str = "sparse"      # sparse | dense | hybrid（M7 起 FTS5 已删除）

    def resolve_milvus_path(self) -> Path:
        p = Path(self.milvus_path)
        return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p


_ALLOWED_RERANK_MODES = ("auto", "off", "api")
_ALLOWED_RERANK_PROTOCOLS = ("jina", "dashscope")
_ALLOWED_RETRIEVERS = ("sparse", "dense", "hybrid")


@dataclass
class LoggingConfig:
    """应用日志（标准库 logging）：默认落盘 ``backend/logs/app.log``，按大小轮转。

    file 相对 ``backend/`` 解析（锚点同 ``RetrievalConfig.resolve_milvus_path``），不随启动
    CWD 漂移；``file: ""`` 表示关闭文件日志、只打控制台。
    ``max_bytes <= 0`` 不轮转；``backups`` 是保留的历史文件数（``app.log.1`` …）。

    已知限制：``RotatingFileHandler`` 不是多进程安全的。服务端已强制单 worker
    （``backend/run.sh --workers 1``：HITL 确认通道、同 thread 串行锁、进程内单例
    Milvus Lite 都依赖它）；需要多 worker 时应改走 journald 或按 pid 分文件。
    """

    level: str = "INFO"
    file: str = "logs/app.log"
    max_bytes: int = 5_000_000    # 单文件上限，超过即轮转
    backups: int = 5              # 保留 app.log.1 … app.log.N
    console: bool = True          # 应用日志是否同时打 stderr（systemd 下进 journald）
    third_party_level: str = "WARNING"   # 压 pymilvus/grpc/httpx 等噪声的级别

    def resolve_file_path(self) -> Path | None:
        """把 file 解析成绝对路径；空串（关闭文件日志）返回 None。"""
        if not self.file.strip():
            return None
        p = Path(self.file)
        return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p


@dataclass
class Settings:
    ocp: OcpConfig = field(default_factory=OcpConfig)
    sql_ro: SqlConfig = field(default_factory=SqlConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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


def _validate_retrieval(settings: Settings) -> None:
    """校验检索相关配置的自洽性（fail fast，只在配置层面，不碰网络/Milvus）。

    设计取舍：embedding 缺配置 **不是** 错误（稀疏一路仍可用，health 暴露降级）；
    只有显式要求稠密一路（default_retriever=dense|hybrid，或 search 传入 dense/hybrid）
    时缺配置才算配置错误。rerank 在 mode=auto 下同样静默降级，mode=api 才 fail closed。
    """
    emb, rk, rt = settings.embedding, settings.rerank, settings.retrieval

    for name, value, low in (
        ("embedding.dims", emb.dims, 1),
        ("embedding.batch_size", emb.batch_size, 1),
        ("embedding.concurrency", emb.concurrency, 1),
        ("embedding.timeout_seconds", emb.timeout_seconds, 1),
        ("embedding.query_cache_size", emb.query_cache_size, 0),
        ("retrieval.pool_k", rt.pool_k, 1),
        ("retrieval.rrf_k", rt.rrf_k, 1),
        ("retrieval.nlist", rt.nlist, 1),
        ("retrieval.query_timeout_seconds", rt.query_timeout_seconds, 1),
        ("retrieval.query_cache_size", rt.query_cache_size, 0),
        ("retrieval.max_text_bytes", rt.max_text_bytes, 1),
        ("retrieval.fingerprint_ttl_seconds", rt.fingerprint_ttl_seconds, 0),
        ("rerank.top_n", rk.top_n, 1),
        ("rerank.max_passage_chars", rk.max_passage_chars, 1),
    ):
        if value < low:
            raise ValueError(f"{name} 必须 >= {low}，当前 {value}")
    if rk.timeout_seconds <= 0:
        raise ValueError(f"rerank.timeout_seconds 必须 > 0，当前 {rk.timeout_seconds}")

    if rk.mode not in _ALLOWED_RERANK_MODES:
        raise ValueError(
            f"rerank.mode 必须是 {list(_ALLOWED_RERANK_MODES)} 之一，当前 {rk.mode!r}"
        )
    if rk.protocol not in _ALLOWED_RERANK_PROTOCOLS:
        raise ValueError(
            f"rerank.protocol 必须是 {list(_ALLOWED_RERANK_PROTOCOLS)} 之一，当前 {rk.protocol!r}"
        )
    if rk.mode == "api" and not rk.is_configured:
        raise ValueError(
            "rerank.mode=api 但 rerank.base_url / rerank.model 未配置齐全："
            "要么补全配置，要么把 mode 改为 auto（缺配置静默关闭）或 off"
        )

    if rt.default_retriever not in _ALLOWED_RETRIEVERS:
        raise ValueError(
            f"retrieval.default_retriever 必须是 {list(_ALLOWED_RETRIEVERS)} 之一，"
            f"当前 {rt.default_retriever!r}"
        )
    if rt.default_retriever in ("dense", "hybrid") and not emb.is_configured:
        raise ValueError(
            f"retrieval.default_retriever={rt.default_retriever} 需要稠密一路，"
            "但 embedding.base_url / embedding.model 未配置齐全"
        )


def _validate_logging(cfg: LoggingConfig) -> None:
    """校验日志配置（fail fast）。

    level/third_party_level 必须是 logging 认识的级别名：写错时 ``setLevel`` 不会报错，
    而是静默失效（等于保持原级别），用户会以为「配了 DEBUG 却没详细日志」，所以在启动时
    就报出来。max_bytes/backups 允许 <=0（不轮转 / 不留备份），不做下界校验。
    """
    known = sorted(logging.getLevelNamesMapping())
    for name, value in (
        ("logging.level", cfg.level),
        ("logging.third_party_level", cfg.third_party_level),
    ):
        if str(value).strip().upper() not in logging.getLevelNamesMapping():
            raise ValueError(f"{name} 不是合法日志级别：{value!r}（可选 {known}）")


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
    retrieval_y = data.get("retrieval", {}) or {}
    embedding_y = data.get("embedding", {}) or {}
    rerank_y = data.get("rerank", {}) or {}
    logging_y = data.get("logging", {}) or {}

    verify_ssl_env = _env_nonempty(env, "OCP_VERIFY_SSL")
    send_row_data_env = _env_nonempty(env, "SEND_ROW_DATA")
    memory_enabled_env = _env_nonempty(env, "MEMORY_ENABLED")

    # file 的空串有语义（关闭文件日志），所以不能像其他字段那样直接 get(..., 默认值)：
    # 0 / "" 都是合法取值，用 `or` 兜底会把它们悄悄换成默认值。
    log_file = _env_nonempty(env, "LOG_FILE")
    if log_file is None:
        log_file_y = logging_y.get("file", "logs/app.log")
        log_file = log_file_y if isinstance(log_file_y, str) else "logs/app.log"
    log_max_bytes_y = logging_y.get("max_bytes")
    log_backups_y = logging_y.get("backups")
    log_console_env = _env_nonempty(env, "LOG_CONSOLE")

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
        retrieval=RetrievalConfig(
            milvus_path=_env_nonempty(env, "RETRIEVAL_MILVUS_PATH")
            or retrieval_y.get("milvus_path", "doc/ob_wiki.milvus.db"),
            collection=_env_nonempty(env, "RETRIEVAL_COLLECTION")
            or retrieval_y.get("collection", "ob_chunks"),
            meta_collection=_env_nonempty(env, "RETRIEVAL_META_COLLECTION")
            or retrieval_y.get("meta_collection", "ob_meta"),
            analyzer=_env_nonempty(env, "RETRIEVAL_ANALYZER") or retrieval_y.get("analyzer", "jieba"),
            dense_index_type=_env_nonempty(env, "RETRIEVAL_DENSE_INDEX_TYPE")
            or retrieval_y.get("dense_index_type", "IVF_FLAT"),
            nlist=int(_env_nonempty(env, "RETRIEVAL_NLIST") or retrieval_y.get("nlist", 128)),
            metric=_env_nonempty(env, "RETRIEVAL_METRIC") or retrieval_y.get("metric", "COSINE"),
            pool_k=int(_env_nonempty(env, "RETRIEVAL_POOL_K") or retrieval_y.get("pool_k", 50)),
            rrf_k=int(_env_nonempty(env, "RETRIEVAL_RRF_K") or retrieval_y.get("rrf_k", 60)),
            weight_sparse=float(
                _env_nonempty(env, "RETRIEVAL_WEIGHT_SPARSE") or retrieval_y.get("weight_sparse", 1.0)
            ),
            weight_dense=float(
                _env_nonempty(env, "RETRIEVAL_WEIGHT_DENSE") or retrieval_y.get("weight_dense", 1.0)
            ),
            query_timeout_seconds=int(
                _env_nonempty(env, "RETRIEVAL_QUERY_TIMEOUT_SECONDS")
                or retrieval_y.get("query_timeout_seconds", 5)
            ),
            query_cache_size=int(
                _env_nonempty(env, "RETRIEVAL_QUERY_CACHE_SIZE") or retrieval_y.get("query_cache_size", 512)
            ),
            max_text_bytes=int(
                _env_nonempty(env, "RETRIEVAL_MAX_TEXT_BYTES") or retrieval_y.get("max_text_bytes", 8000)
            ),
            fingerprint_ttl_seconds=float(
                _env_nonempty(env, "RETRIEVAL_FINGERPRINT_TTL_SECONDS")
                or retrieval_y.get("fingerprint_ttl_seconds", 5.0)
            ),
            version_match_bonus=float(
                _env_nonempty(env, "RETRIEVAL_VERSION_MATCH_BONUS")
                or retrieval_y.get("version_match_bonus", 8.0)
            ),
            nav_section_penalty=float(
                _env_nonempty(env, "RETRIEVAL_NAV_SECTION_PENALTY")
                or retrieval_y.get("nav_section_penalty", 12.0)
            ),
            nav_file_penalty=float(
                _env_nonempty(env, "RETRIEVAL_NAV_FILE_PENALTY") or retrieval_y.get("nav_file_penalty", 40.0)
            ),
            default_retriever=_env_nonempty(env, "RETRIEVAL_DEFAULT_RETRIEVER")
            or retrieval_y.get("default_retriever", "sparse"),
        ),
        embedding=EmbeddingConfig(
            base_url=_env_nonempty(env, "EMBEDDING_BASE_URL") or embedding_y.get("base_url", ""),
            api_key=_env_nonempty(env, "EMBEDDING_API_KEY") or embedding_y.get("api_key", ""),
            model=_env_nonempty(env, "EMBEDDING_MODEL") or embedding_y.get("model", ""),
            dims=int(_env_nonempty(env, "EMBEDDING_DIMS") or embedding_y.get("dims", 1024)),
            batch_size=int(_env_nonempty(env, "EMBEDDING_BATCH_SIZE") or embedding_y.get("batch_size", 64)),
            concurrency=int(
                _env_nonempty(env, "EMBEDDING_CONCURRENCY") or embedding_y.get("concurrency", 4)
            ),
            timeout_seconds=int(
                _env_nonempty(env, "EMBEDDING_TIMEOUT_SECONDS") or embedding_y.get("timeout_seconds", 30)
            ),
            query_cache_size=int(
                _env_nonempty(env, "EMBEDDING_QUERY_CACHE_SIZE") or embedding_y.get("query_cache_size", 512)
            ),
        ),
        rerank=RerankConfig(
            mode=_env_nonempty(env, "RERANK_MODE") or rerank_y.get("mode", "auto"),
            protocol=_env_nonempty(env, "RERANK_PROTOCOL") or rerank_y.get("protocol", "jina"),
            base_url=_env_nonempty(env, "RERANK_BASE_URL") or rerank_y.get("base_url", ""),
            api_key=_env_nonempty(env, "RERANK_API_KEY") or rerank_y.get("api_key", ""),
            model=_env_nonempty(env, "RERANK_MODEL") or rerank_y.get("model", ""),
            top_n=int(_env_nonempty(env, "RERANK_TOP_N") or rerank_y.get("top_n", 30)),
            timeout_seconds=float(
                _env_nonempty(env, "RERANK_TIMEOUT_SECONDS") or rerank_y.get("timeout_seconds", 3.0)
            ),
            max_passage_chars=int(
                _env_nonempty(env, "RERANK_MAX_PASSAGE_CHARS") or rerank_y.get("max_passage_chars", 500)
            ),
        ),
        logging=LoggingConfig(
            level=_env_nonempty(env, "LOG_LEVEL") or logging_y.get("level", "INFO"),
            file=log_file,
            max_bytes=int(
                _env_nonempty(env, "LOG_MAX_BYTES")
                or (5_000_000 if log_max_bytes_y is None else log_max_bytes_y)
            ),
            backups=int(
                _env_nonempty(env, "LOG_BACKUPS") or (5 if log_backups_y is None else log_backups_y)
            ),
            console=_as_bool(
                log_console_env if log_console_env is not None else logging_y.get("console"),
                True,
            ),
            third_party_level=_env_nonempty(env, "LOG_THIRD_PARTY_LEVEL")
            or logging_y.get("third_party_level", "WARNING"),
        ),
    )
    _validate_agent(settings.agent)
    _validate_retrieval(settings)
    _validate_logging(settings.logging)
    return settings