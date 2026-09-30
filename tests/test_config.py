from pathlib import Path

import pytest

from app.config import load_settings

# 「没有配置文件」的默认值用例必须显式指向一个不存在的路径：
# load_settings(config_path=None) 会回退到 backend/config.yaml，而开发机/部署机上
# 这个文件通常是存在的（里面有真实 OCP/SQL/LLM 配置），传 None 会让断言随环境漂移。
_MISSING_CONFIG = "/nonexistent-ob-agent-test-config.yaml"


def _write_yaml(tmp_path: Path, text: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_defaults_are_mock_when_no_file(tmp_path):
    s = load_settings(config_path=_MISSING_CONFIG, env={"OCP_PROVIDER": "mock"})
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


# ---- 审批超时必须严格短于整轮上限（否则审批超时永远不会先触发）----------------


def test_default_confirm_timeout_is_strictly_below_max_seconds():
    s = load_settings(config_path=_MISSING_CONFIG, env={"OCP_PROVIDER": "mock"})
    assert 0 < s.agent.confirm_timeout_seconds < s.agent.max_seconds


def test_confirm_timeout_not_less_than_max_seconds_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        "agent:\n  confirm_db_ops: true\n  max_seconds: 60\n  confirm_timeout_seconds: 60\n",
    )
    with pytest.raises(ValueError):
        load_settings(config_path=path, env={})


def test_confirm_timeout_equal_to_max_is_allowed_when_hitl_disabled(tmp_path):
    path = _write_yaml(
        tmp_path,
        "agent:\n  confirm_db_ops: false\n  max_seconds: 60\n  confirm_timeout_seconds: 60\n",
    )
    s = load_settings(config_path=path, env={})
    assert s.agent.confirm_timeout_seconds == 60


def test_confirm_timeout_must_be_positive(tmp_path):
    path = _write_yaml(tmp_path, "agent:\n  confirm_timeout_seconds: 0\n")
    with pytest.raises(ValueError):
        load_settings(config_path=path, env={})


# ---- 检索 / embedding / rerank（先让用户能填 key，再接线到 M2/M3）----------------


def test_retrieval_defaults_when_no_file():
    s = load_settings(config_path=_MISSING_CONFIG, env={"OCP_PROVIDER": "mock"})
    assert s.retrieval.milvus_path == "doc/ob_wiki.milvus.db"
    assert s.retrieval.collection == "ob_chunks"
    assert s.retrieval.meta_collection == "ob_meta"
    assert s.retrieval.analyzer == "jieba"
    assert s.retrieval.rrf_k == 60
    assert s.retrieval.pool_k == 50
    # M7 删掉 FTS5 后默认就是 Milvus 稀疏（唯一过门禁的一路）
    assert s.retrieval.default_retriever == "sparse"
    assert s.embedding.is_configured is False
    assert s.rerank.mode == "auto"
    assert s.rerank.enabled is False


def test_retrieval_sections_are_loaded_from_yaml(tmp_path):
    path = _write_yaml(
        tmp_path,
        "retrieval:\n  pool_k: 20\n  rrf_k: 10\n"
        "embedding:\n  base_url: http://e/v1\n  model: m\n  dims: 768\n"
        "rerank:\n  mode: api\n  base_url: http://r/v1\n  model: rr\n",
    )
    s = load_settings(config_path=path, env={})
    assert s.retrieval.pool_k == 20
    assert s.retrieval.rrf_k == 10
    assert s.embedding.dims == 768
    assert s.embedding.endpoint_url == "http://e/v1/embeddings"
    assert s.rerank.enabled is True
    assert s.rerank.endpoint_url == "http://r/v1/rerank"


def test_retrieval_env_overrides_yaml(tmp_path):
    path = _write_yaml(tmp_path, "embedding:\n  base_url: http://yaml/v1\n  model: m\n")
    s = load_settings(
        config_path=path,
        env={"EMBEDDING_BASE_URL": "http://env/v1", "EMBEDDING_API_KEY": "k"},
    )
    assert s.embedding.base_url == "http://env/v1"
    assert s.embedding.api_key == "k"


def test_retrieval_env_empty_string_does_not_clobber_yaml(tmp_path):
    path = _write_yaml(
        tmp_path,
        "embedding:\n  base_url: http://e/v1\n  model: m\n  api_key: from_yaml\n"
        "rerank:\n  mode: api\n  base_url: http://r/v1\n  model: rr\n  api_key: from_yaml\n",
    )
    s = load_settings(config_path=path, env={"EMBEDDING_API_KEY": "", "RERANK_API_KEY": ""})
    assert s.embedding.api_key == "from_yaml"
    assert s.rerank.api_key == "from_yaml"


def test_embedding_is_configured_does_not_require_api_key():
    from app.config import EmbeddingConfig

    # 本地 Ollama/vLLM 免 key：缺 api_key 不能判成「未配置」，否则本地开发会被禁用稠密一路
    assert EmbeddingConfig(base_url=" http://e/v1 ", model="m").is_configured is True
    assert EmbeddingConfig(base_url="http://e/v1").is_configured is False
    assert EmbeddingConfig(model="m").is_configured is False
    assert EmbeddingConfig().is_configured is False


def test_rerank_api_mode_without_config_raises(tmp_path):
    path = _write_yaml(tmp_path, "rerank:\n  mode: api\n")
    with pytest.raises(ValueError):
        load_settings(config_path=path, env={})


def test_rerank_auto_without_config_stays_off(tmp_path):
    path = _write_yaml(tmp_path, "rerank:\n  mode: auto\n")
    s = load_settings(config_path=path, env={})
    assert s.rerank.enabled is False


def test_rerank_invalid_mode_raises(tmp_path):
    path = _write_yaml(tmp_path, "rerank:\n  mode: sometimes\n")
    with pytest.raises(ValueError):
        load_settings(config_path=path, env={})


def test_default_retriever_needs_embedding_only_for_dense(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("retrieval:\n  default_retriever: hybrid\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(config_path=str(bad), env={})

    good = tmp_path / "good.yaml"
    good.write_text(
        "retrieval:\n  default_retriever: hybrid\n"
        "embedding:\n  base_url: http://e/v1\n  model: m\n",
        encoding="utf-8",
    )
    s = load_settings(config_path=str(good), env={})
    assert s.retrieval.default_retriever == "hybrid"

    # 只跑稀疏一路时不要求 embedding 配置
    sparse = tmp_path / "sparse.yaml"
    sparse.write_text("retrieval:\n  default_retriever: sparse\n", encoding="utf-8")
    assert load_settings(config_path=str(sparse), env={}).retrieval.default_retriever == "sparse"


def test_default_retriever_invalid_value_raises(tmp_path):
    path = _write_yaml(tmp_path, "retrieval:\n  default_retriever: bm25\n")
    with pytest.raises(ValueError):
        load_settings(config_path=path, env={})


def test_embedding_dims_must_be_positive(tmp_path):
    path = _write_yaml(tmp_path, "embedding:\n  dims: 0\n")
    with pytest.raises(ValueError):
        load_settings(config_path=path, env={})


def test_resolve_milvus_path_is_anchored_to_backend_dir():
    from app.config import RetrievalConfig

    p = RetrievalConfig().resolve_milvus_path()
    assert p.is_absolute()
    assert p == Path(__file__).resolve().parent.parent / "backend" / "doc" / "ob_wiki.milvus.db"


# ---- rerank 协议形态（实测：阿里云百炼 compatible-mode/v1 下没有 /rerank，只有原生形态）----


def test_rerank_jina_endpoint_appends_rerank():
    from app.config import RerankConfig

    cfg = RerankConfig(base_url="https://jina.ai/v1/", model="m")
    assert cfg.protocol == "jina"
    assert cfg.endpoint_url == "https://jina.ai/v1/rerank"


def test_rerank_dashscope_endpoint_is_derived_from_host():
    from app.config import RerankConfig

    cfg = RerankConfig(
        protocol="dashscope",
        base_url="https://ws-x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-text-rerank",
    )
    assert cfg.endpoint_url == (
        "https://ws-x.cn-beijing.maas.aliyuncs.com"
        "/api/v1/services/rerank/text-rerank/text-rerank"
    )
    # 已经是原生路径时原样返回，不再拼接
    native = RerankConfig(
        protocol="dashscope",
        base_url="https://h/api/v1/services/rerank/text-rerank/text-rerank",
        model="m",
    )
    assert native.endpoint_url == native.base_url


def test_rerank_protocol_loaded_from_yaml_and_validated(tmp_path):
    path = _write_yaml(
        tmp_path,
        "rerank:\n  mode: api\n  protocol: dashscope\n"
        "  base_url: https://h/compatible-mode/v1\n  model: m\n",
    )
    assert load_settings(config_path=path, env={}).rerank.protocol == "dashscope"

    bad = _write_yaml(tmp_path, "rerank:\n  protocol: cohere\n")
    with pytest.raises(ValueError):
        load_settings(config_path=bad, env={})


def test_embedding_batch_size_default_is_under_provider_limit():
    from app.config import EmbeddingConfig

    # DashScope 兼容模式硬上限 20（超限 400），默认必须留余量
    assert EmbeddingConfig().batch_size <= 20


# ---- 应用日志（M9：日志持久化）----


def test_logging_defaults_are_set():
    from app.config import LoggingConfig

    cfg = LoggingConfig()
    assert cfg.level == "INFO"
    assert cfg.file == "logs/app.log"
    assert cfg.max_bytes == 5_000_000
    assert cfg.backups == 5
    assert cfg.console is True
    assert cfg.third_party_level == "WARNING"


def test_logging_section_loaded_from_yaml(tmp_path):
    path = _write_yaml(
        tmp_path,
        "logging:\n  file: logs/other.log\n  level: DEBUG\n  max_bytes: 1024\n"
        "  backups: 2\n  console: false\n  third_party_level: ERROR\n",
    )
    cfg = load_settings(config_path=path, env={}).logging
    assert cfg.file == "logs/other.log"
    assert cfg.level == "DEBUG"
    assert cfg.max_bytes == 1024
    assert cfg.backups == 2
    assert cfg.console is False
    assert cfg.third_party_level == "ERROR"


def test_logging_env_overrides_yaml(tmp_path):
    path = _write_yaml(tmp_path, "logging:\n  file: logs/other.log\n  level: DEBUG\n")
    cfg = load_settings(
        config_path=path,
        env={"LOG_FILE": "/tmp/env.log", "LOG_LEVEL": "warning", "LOG_CONSOLE": "false"},
    ).logging
    assert cfg.file == "/tmp/env.log"
    assert cfg.level == "warning"      # 大小写不敏感，启动时统一 upper 校验
    assert cfg.console is False


def test_logging_env_empty_string_does_not_clobber_yaml(tmp_path):
    path = _write_yaml(tmp_path, "logging:\n  file: logs/other.log\n  level: DEBUG\n")
    cfg = load_settings(config_path=path, env={"LOG_FILE": "", "LOG_LEVEL": ""}).logging
    assert cfg.file == "logs/other.log"
    assert cfg.level == "DEBUG"


def test_logging_level_validation_rejects_unknown(tmp_path):
    path = _write_yaml(tmp_path, "logging:\n  level: verbose\n")
    with pytest.raises(ValueError) as ei:
        load_settings(config_path=path, env={})
    assert "不是合法日志级别" in str(ei.value)


def test_logging_unknown_env_level_is_rejected():
    # setLevel 对写错的级别名静默失效，所以必须在启动时 fail fast
    with pytest.raises(ValueError):
        load_settings(config_path=_MISSING_CONFIG, env={"LOG_LEVEL": "nonsense"})


def test_logging_blank_file_disables_file_handler(tmp_path):
    path = _write_yaml(tmp_path, "logging:\n  file: ''\n")
    assert load_settings(config_path=path, env={}).logging.resolve_file_path() is None


def test_logging_zero_limits_are_preserved(tmp_path):
    """0 表示「不轮转 / 不留备份」，不能被 `or` 兜底成默认值。"""
    path = _write_yaml(tmp_path, "logging:\n  max_bytes: 0\n  backups: 0\n")
    cfg = load_settings(config_path=path, env={}).logging
    assert cfg.max_bytes == 0
    assert cfg.backups == 0


def test_resolve_log_path_is_anchored_to_backend_dir():
    from app.config import LoggingConfig

    p = LoggingConfig(file="logs/app.log").resolve_file_path()
    assert p is not None and p.is_absolute()
    assert p == Path(__file__).resolve().parent.parent / "backend" / "logs" / "app.log"
