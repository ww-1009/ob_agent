"""Milvus Lite 索引层：集合 schema、连接生命周期、``ob_meta`` 读写、启动校验。

为什么单独一个模块：

- ``doc_index.py`` 是 FTS5 的旧世界（M7 删除），``retrieval.py``（M5）是查询期融合；两边都要
  「同一个集合定义」「同一份元数据」。放这里，schema 只写一次，不会漂移。
- milvus-lite 3.2.1 是**纯 Python 进程内实现**（``milvus_lite/server_manager.py`` 起本地 gRPC
  线程，随机端口），``data_dir/LOCK`` 是 advisory ``flock``：**一个目录同时只能被一个进程持有**。
  更坑的是 pymilvus 全仓没有 ``release_server`` 调用，``MilvusClient.close()`` **不释放锁**，
  锁活到进程退出。于是这里负责把这两个坑翻译成清晰异常，并在需要交接目录时显式调
  ``ServerManager.release_server()``（实测有效，见 probe7 第 5 节）。
- 单进程约束还有两个直接后果：``uvicorn --workers 1``（``run.sh`` 已单 worker），以及
  **建索引与在线服务不能同时打开同一个 data_dir**（重建走「临时目录 → 目录级切换」）。

字段口径见 ``backend/eval/docs/milvus-only-retrieval-design.md`` §5.1：

- ``pk``：``xxh64(f"{path}#{section}#{seq}")``，用**身份**做键而不是内容哈希（两篇文档可以有完全相同的段落）。
  带 ``seq`` 是因为 ``_split_chunks`` 会把超长小节切成多块：现有库里有 1630 组 ``(path, section)`` 重复，
  单组最多 29 块，只按 ``path#section`` 做键会在 upsert 时互相覆盖。
- ``text``：被 BM25 分析的文本（标题前缀重复 + 小节 + 正文），也是 snippet 来源。
- ``vector``：``FLOAT_VECTOR(dims)``。**不可空**——省略字段或显式 None 都会被拒（probe7 第 1 节），
  导航行因此写零向量，稠密一路恒 ``filter kind == "doc"``（零向量在 COSINE 下 distance=0，不会上浮）。
- ``kind``：``doc`` / ``nav``。导航行**照常写入**（v1 §8.4 的 −40 惩罚与 ``tests/test_retrieval_eval.py``
  的「导航页永不 top1」硬门禁都依赖它），只是不进稠密一路。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pymilvus import (
    CollectionSchema,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
)

from app.config import EmbeddingConfig, RetrievalConfig, Settings, load_settings

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# ---- ob_chunks 字段名（全项目只在这里定义一次）----
FIELD_PK = "pk"
FIELD_CONTENT_HASH = "content_hash"
FIELD_TEXT = "text"
FIELD_SPARSE = "sparse"
FIELD_VECTOR = "vector"
FIELD_KIND = "kind"
FIELD_PATH = "path"
FIELD_SECTION = "section"
FIELD_TITLE = "title"
FIELD_MODE = "mode"
FIELD_VERSION = "version"

BM25_FUNCTION = "bm25"

# ---- ob_meta ----
META_FIELD_KEY = "key"
META_FIELD_VALUE = "value"
# milvus-lite 不允许「没有向量字段」的集合（probe7 第 7 节：
# MilvusException(code=6, message=schema has no vector field)），元数据集合挂一个 2 维占位向量。
META_DUMMY_FIELD = "_dummy"
META_DUMMY_DIM = 2

META_SCHEMA_VERSION = "schema_version"
META_CORPUS_FINGERPRINT = "corpus_fingerprint"
META_EMBEDDING_MODEL = "embedding_model"
META_DIMS = "dims"
META_ANALYZER = "analyzer"
META_BUILT_AT = "built_at"
META_TEXT_MAX_LENGTH = "text_max_length"

# VARCHAR 长度（milvus-lite 3.2.1 实测按**字符**计，不是字节；probe7 第 2 节：
# max_length=100 接受 100 个汉字 = 300 字节）。正文上限见 RetrievalConfig.max_text_bytes 守卫。
TEXT_MAX_LENGTH = 8000
PATH_MAX_LENGTH = 512
SECTION_MAX_LENGTH = 512
TITLE_MAX_LENGTH = 512
MODE_MAX_LENGTH = 16
VERSION_MAX_LENGTH = 16
KIND_MAX_LENGTH = 8
CONTENT_HASH_MAX_LENGTH = 32

# FAISS 的 IVF 训练需要足够的点：经验值是 39×nlist。25k 块 / nlist=128 够用（4992 点），
# 但合成小语料（测试里的 5~50 块）会直接训练失败，所以小集合要退回 FLAT/BRUTE_FORCE。
IVF_TRAINING_RATIO = 39


class MilvusUnavailable(RuntimeError):
    """Milvus 打不开 / 查询失败。调用方降级为「空结果 + retrieval_degraded」，不要抛 500。"""


class MilvusIndexMissing(MilvusUnavailable):
    """data_dir 或集合还不存在：需要先跑 ``python -m app.agent.milvus_index --rebuild``。"""


class MilvusSchemaMismatch(MilvusUnavailable):
    """``ob_meta`` 与当前配置不一致（schema_version / analyzer / dims）：需要重建。"""


@dataclass
class ChunkRow:
    """写入 ``ob_chunks`` 的一行（向量由 embedding 客户端拿到后填入）。"""

    pk: int
    text: str
    kind: str
    path: str
    section: str = ""
    title: str = ""
    mode: str = ""
    version: str = ""
    content_hash: str = ""
    vector: Sequence[float] | None = None

    def to_entity(self, *, zero_vector: Sequence[float] | None = None) -> dict[str, Any]:
        """转成 ``insert``/``upsert`` 的字典。

        ``vector`` 为 None 时用 ``zero_vector`` 兜底——导航行没有稠密向量，但 FLOAT_VECTOR
        不可空（probe7 第 1 节），零向量是唯一能写进去的占位。
        """
        vector = self.vector if self.vector is not None else zero_vector
        if vector is None:
            raise MilvusUnavailable(f"pk={self.pk} 缺少向量且未提供零向量兜底")
        return {
            FIELD_PK: int(self.pk),
            FIELD_CONTENT_HASH: self.content_hash,
            FIELD_TEXT: self.text,
            FIELD_VECTOR: list(vector),
            FIELD_KIND: self.kind,
            FIELD_PATH: self.path,
            FIELD_SECTION: self.section,
            FIELD_TITLE: self.title,
            FIELD_MODE: self.mode,
            FIELD_VERSION: self.version,
        }


# ---------------------------------------------------------------- schema


def chunks_schema(*, dims: int, text_max_length: int = TEXT_MAX_LENGTH) -> CollectionSchema:
    """``ob_chunks`` 的 schema（含 BM25 Function）。"""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(FIELD_PK, DataType.INT64, is_primary=True)
    schema.add_field(FIELD_CONTENT_HASH, DataType.VARCHAR, max_length=CONTENT_HASH_MAX_LENGTH)
    schema.add_field(
        FIELD_TEXT,
        DataType.VARCHAR,
        max_length=text_max_length,
        enable_analyzer=True,
        analyzer_params={"type": "jieba"},
    )
    schema.add_field(FIELD_SPARSE, DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(FIELD_VECTOR, DataType.FLOAT_VECTOR, dim=int(dims))
    schema.add_field(FIELD_KIND, DataType.VARCHAR, max_length=KIND_MAX_LENGTH)
    schema.add_field(FIELD_PATH, DataType.VARCHAR, max_length=PATH_MAX_LENGTH)
    schema.add_field(FIELD_SECTION, DataType.VARCHAR, max_length=SECTION_MAX_LENGTH)
    schema.add_field(FIELD_TITLE, DataType.VARCHAR, max_length=TITLE_MAX_LENGTH)
    schema.add_field(FIELD_MODE, DataType.VARCHAR, max_length=MODE_MAX_LENGTH)
    schema.add_field(FIELD_VERSION, DataType.VARCHAR, max_length=VERSION_MAX_LENGTH)
    schema.add_function(
        Function(
            name=BM25_FUNCTION,
            function_type=FunctionType.BM25,
            input_field_names=[FIELD_TEXT],
            output_field_names=[FIELD_SPARSE],
        )
    )
    return schema


def dense_index_type_for(rows: int, config: RetrievalConfig) -> str:
    """小集合退回 FLAT：IVF 训练点不够时 faiss 会直接报错。

    25k 块 + ``nlist=128`` 用配置里的 ``IVF_FLAT``；测试里的合成小语料（几块到几十块）走 FLAT
    （milvus-lite 把 FLAT/BRUTE_FORCE 都映射到无需训练的实现）。
    """
    if rows < IVF_TRAINING_RATIO * max(1, int(config.nlist)):
        return "FLAT"
    return config.dense_index_type


def chunks_index_params(
    client: MilvusClient,
    *,
    config: RetrievalConfig,
    rows: int | None = None,
) -> Any:
    """``ob_chunks`` 的索引声明：稀疏 BM25 + 稠密 + 三个标量倒排。"""
    params = client.prepare_index_params()
    params.add_index(FIELD_SPARSE, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    dense_type = dense_index_type_for(rows, config) if rows is not None else config.dense_index_type
    dense_params: dict[str, Any] = {}
    if dense_type == "IVF_FLAT":
        dense_params["nlist"] = int(config.nlist)
    params.add_index(FIELD_VECTOR, index_type=dense_type, metric_type=config.metric, params=dense_params)
    for name in (FIELD_KIND, FIELD_MODE, FIELD_VERSION, FIELD_PATH):
        params.add_index(name, index_type="INVERTED")
    return params


def meta_schema() -> CollectionSchema:
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(META_FIELD_KEY, DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field(META_FIELD_VALUE, DataType.VARCHAR, max_length=512)
    schema.add_field(META_DUMMY_FIELD, DataType.FLOAT_VECTOR, dim=META_DUMMY_DIM)
    return schema


def meta_index_params(client: MilvusClient) -> Any:
    params = client.prepare_index_params()
    params.add_index(META_DUMMY_FIELD, index_type="FLAT", metric_type="L2")
    return params


def meta_entity(key: str, value: Any) -> dict[str, Any]:
    return {META_FIELD_KEY: key, META_FIELD_VALUE: str(value), META_DUMMY_FIELD: [0.0] * META_DUMMY_DIM}


# ---------------------------------------------------------------- 连接


def _lock_error_hint(path: Path) -> str:
    return (
        f"Milvus data_dir 打不开：{path}。"
        "milvus-lite 用 data_dir/LOCK 做进程级排他（一个目录一个进程），"
        "常见原因是已有进程持有该目录（开发服务/索引构建/上一个未退出的测试），"
        "另外 uvicorn 必须以 --workers 1 运行、不要开 --reload。"
    )


def open_client(path: Path | str) -> MilvusClient:
    """打开（必要时创建）data_dir。失败统一抛 ``MilvusUnavailable``（带可操作提示）。"""
    target = Path(path)
    # milvus-lite 只认「以 .db 结尾」的本地路径（M4 实测：plain.milvus / noext 都被拒），
    # 配置里写错后缀会得到一句很绕的 ConnectionConfigException，这里提前说清楚。
    if target.suffix != ".db":
        raise MilvusUnavailable(
            f"Milvus data_dir 路径必须以 .db 结尾（milvus-lite 的硬要求）：{target}"
        )
    if not target.parent.is_dir():
        raise MilvusUnavailable(f"Milvus data_dir 的父目录不存在：{target.parent}")
    try:
        return MilvusClient(str(target))
    except Exception as exc:  # noqa: BLE001 - 底层异常类型随 pymilvus 版本变化
        raise MilvusUnavailable(f"{_lock_error_hint(target)}（原始错误：{type(exc).__name__}: {exc}）") from exc


def release_server(path: Path | str) -> None:
    """释放当前进程对 data_dir 的持有。

    ``MilvusClient.close()`` 不够：pymilvus 从不调用 ``release_server``，锁会一直握到进程退出。
    索引构建完成后要把目录交还给在线服务（或反之），必须显式调这个（probe7 第 5 节实测有效）。
    """
    try:
        from milvus_lite.server_manager import server_manager_instance
    except ImportError:  # pragma: no cover - 没有 milvus-lite 时无所谓
        return
    server_manager_instance.release_server(str(Path(path).absolute()))


def probe_and_release(path: Path | str) -> str:
    """不长期持有目录地探一次「能不能打开」。

    返回 ``"missing"``（目录不存在，需先建索引）/ ``"ok"``；被别的进程占用时抛 ``MilvusUnavailable``。
    """
    target = Path(path)
    if not target.is_dir():
        return "missing"
    client = open_client(target)
    try:
        client.list_collections()
    finally:
        client.close()
        release_server(target)
    return "ok"


# ---------------------------------------------------------------- 索引对象


@dataclass
class MilvusIndex:
    """长生命周期的 Milvus 句柄（进程内单例，由 ``get_milvus_index()`` 复用）。

    M3 只做「连接 + 建集合 + 元数据 + 统计」；索引构建/增量在 M4，查询融合在 M5。
    """

    config: RetrievalConfig
    dims: int = 1024
    embedding_model: str = ""
    #: 指向别处（M4 的临时构建目录）；None 时用 ``config.resolve_milvus_path()``。
    path_override: Path | str | None = None
    _client: MilvusClient | None = field(default=None, repr=False, compare=False)
    _loaded: set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def path(self) -> Path:
        if self.path_override is not None:
            return Path(self.path_override)
        return self.config.resolve_milvus_path()

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = open_client(self.path)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                # close() 不释放 LOCK：显式交接，避免同一进程内重建时自己把自己锁住。
                release_server(self.path)
                self._client = None
                self._loaded.clear()

    def _ensure_loaded(self, name: str) -> None:
        """``load_collection`` 之后才能 query/search。

        踩坑记录（M3 实测）：**重新打开一个已存在的 data_dir 后集合是 released 状态**，不 load
        直接 query 会得到 ``code=101: Collection 'ob_meta' is in state 'released'``。建集合时
        pymilvus 会自动 load，所以这条规则只在「进程重启 / 同一进程内重开」时才显形——恰好就是
        「索引构建完 → 在线服务打开」这条路径，因此所有读路径都必须过这里。
        """
        if name in self._loaded:
            return
        names = set(self.client.list_collections())
        if name not in names:
            return
        try:
            self.client.load_collection(name)
        except Exception as exc:  # noqa: BLE001
            raise MilvusUnavailable(f"集合 {name} 加载失败：{type(exc).__name__}: {exc}") from exc
        self._loaded.add(name)

    # ---- 查询入口（统一在这里过 load；M5 的两条召回路径直接用这两个包装）----

    def search(
        self,
        data: Any,
        anns_field: str,
        limit: int,
        *,
        filter: str = "",
        output_fields: Sequence[str] | None = None,
        search_params: Mapping[str, Any] | None = None,
        collection: str | None = None,
    ) -> list[Any]:
        name = collection or self.config.collection
        self._ensure_loaded(name)
        kwargs: dict[str, Any] = {"data": data, "anns_field": anns_field, "limit": int(limit)}
        if filter:
            kwargs["filter"] = filter
        if output_fields:
            kwargs["output_fields"] = list(output_fields)
        if search_params:
            kwargs["search_params"] = dict(search_params)
        return self.client.search(name, **kwargs)

    def query(
        self,
        *,
        filter: str = "",
        output_fields: Sequence[str] | None = None,
        limit: int | None = None,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        name = collection or self.config.collection
        self._ensure_loaded(name)
        kwargs: dict[str, Any] = {"filter": filter}
        if output_fields:
            kwargs["output_fields"] = list(output_fields)
        if limit is not None:
            kwargs["limit"] = int(limit)
        return self.client.query(name, **kwargs)

    def query_iterator(
        self,
        *,
        filter: str = "",
        output_fields: Sequence[str] | None = None,
        batch_size: int = 1000,
        collection: str | None = None,
    ) -> Any:
        """全库遍历（``query`` 单次最多 16384 条，枚举旧 pk 做增量剪枝必须用迭代器）。"""
        name = collection or self.config.collection
        self._ensure_loaded(name)
        return self.client.query_iterator(
            name, batch_size=int(batch_size), filter=filter, output_fields=list(output_fields or [])
        )

    def upsert(
        self, entities: Sequence[Mapping[str, Any]], *, collection: str | None = None
    ) -> Any:
        """写入/覆盖行（M4 构建用）。``upsert`` 的每一行都必须带向量。"""
        if not entities:
            return None
        name = collection or self.config.collection
        self._ensure_loaded(name)
        return self.client.upsert(name, list(entities))

    def delete(self, expr: str, *, collection: str | None = None) -> Any:
        """按表达式删除（M4 剪枝用，形如 ``pk in [1, 2]``）。"""
        if not expr:
            return None
        name = collection or self.config.collection
        self._ensure_loaded(name)
        return self.client.delete(name, filter=expr)

    def exists(self) -> bool:
        return self.path.is_dir()

    def has_collections(self) -> bool:
        try:
            names = set(self.client.list_collections())
        except Exception as exc:  # noqa: BLE001
            raise MilvusUnavailable(f"Milvus 集合列表读取失败：{type(exc).__name__}: {exc}") from exc
        return self.config.collection in names and self.config.meta_collection in names

    # ---- 集合与元数据 ----

    def ensure_collections(self, *, create: bool = False, rows: int | None = None) -> dict[str, str]:
        """确保集合存在（``create=True`` 时创建），并校验 ``ob_meta`` 与当前配置一致。

        刚创建（``created=True``）时 ``ob_meta`` 必然为空，那是正常中间态，跳过校验；
        调用方随后应写 ``ob_meta``（M4 的构建流程第 6 步）。集合本来就存在却读不到元数据，
        才说明索引损坏，必须抛。
        """
        client = self.client
        names = set(client.list_collections())
        created = False
        if self.config.collection not in names:
            if not create:
                raise MilvusIndexMissing(
                    f"集合 {self.config.collection!r} 不存在：请先构建索引"
                    "（python -m app.agent.milvus_index --rebuild）"
                )
            client.create_collection(
                self.config.collection,
                schema=chunks_schema(dims=self.dims, text_max_length=self.config.max_text_bytes),
                index_params=chunks_index_params(client, config=self.config, rows=rows),
            )
            client.load_collection(self.config.collection)
            created = True
            logger.info("已创建集合 %s（dims=%d）", self.config.collection, self.dims)
        if self.config.meta_collection not in names:
            if not create:
                raise MilvusIndexMissing(f"集合 {self.config.meta_collection!r} 不存在：请先构建索引")
            client.create_collection(
                self.config.meta_collection,
                schema=meta_schema(),
                index_params=meta_index_params(client),
            )
            client.load_collection(self.config.meta_collection)
            created = True
            logger.info("已创建集合 %s", self.config.meta_collection)
        meta = self.read_meta()
        if meta or not created:
            self._check_meta(meta)
        return meta

    def read_meta(self) -> dict[str, str]:
        self._ensure_loaded(self.config.meta_collection)
        client = self.client
        try:
            rows = client.query(
                self.config.meta_collection,
                filter=f'{META_FIELD_KEY} != ""',
                output_fields=[META_FIELD_KEY, META_FIELD_VALUE],
                limit=64,
            )
        except Exception as exc:  # noqa: BLE001
            raise MilvusUnavailable(f"ob_meta 读取失败：{type(exc).__name__}: {exc}") from exc
        return {row[META_FIELD_KEY]: row[META_FIELD_VALUE] for row in rows}

    def write_meta(self, values: Mapping[str, Any]) -> None:
        """整体覆盖式写入（``upsert``），不删除未列出的键。"""
        if not values:
            return
        entities = [meta_entity(k, v) for k, v in values.items()]
        self.client.upsert(self.config.meta_collection, entities)

    def _check_meta(self, meta: Mapping[str, str]) -> None:
        if not meta:
            raise MilvusSchemaMismatch(
                f"{self.config.meta_collection} 为空：索引元数据缺失，请重建"
                "（python -m app.agent.milvus_index --rebuild）"
            )
        version = str(meta.get(META_SCHEMA_VERSION, ""))
        if version and version != str(SCHEMA_VERSION):
            raise MilvusSchemaMismatch(f"schema_version={version} 与当前 {SCHEMA_VERSION} 不一致：请重建索引")
        analyzer = str(meta.get(META_ANALYZER, ""))
        if analyzer and analyzer != self.config.analyzer:
            raise MilvusSchemaMismatch(
                f"analyzer={analyzer!r} 与配置 {self.config.analyzer!r} 不一致：换分词器等于换检索口径，必须重建并重测基线"
            )
        dims = str(meta.get(META_DIMS, ""))
        if dims and dims.isdigit() and int(dims) != int(self.dims):
            raise MilvusSchemaMismatch(
                f"ob_meta.dims={dims} 与 embedding.dims={self.dims} 不一致：请用 --rebuild-vectors 重算向量"
            )

    # ---- 统计与健康 ----

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "path": str(self.path),
            "collection": self.config.collection,
            "exists": self.exists(),
        }
        # 存在性检查必须在取 client 之前：MilvusClient(path) 会**顺手把目录建出来**，
        # 于是「未建索引」会被伪装成「已存在」，运维排查时非常误导。
        if not self.exists():
            return out
        client = self.client
        self._ensure_loaded(self.config.collection)
        try:
            out["rows"] = int(client.get_collection_stats(self.config.collection).get("row_count", 0))
        except Exception as exc:  # noqa: BLE001
            out["rows"] = None
            out["rows_error"] = f"{type(exc).__name__}: {exc}"
        for label, expr in (("doc_rows", f'{FIELD_KIND} == "doc"'), ("nav_rows", f'{FIELD_KIND} == "nav"')):
            try:
                out[label] = self.count(expr)
            except Exception as exc:  # noqa: BLE001 - count(*) 在 Lite 上的支持待 M4 验证
                out[label] = None
                out[label + "_error"] = f"{type(exc).__name__}: {exc}"
        try:
            index = client.describe_index(self.config.collection, FIELD_VECTOR)
            out["dense_index"] = {k: index.get(k) for k in ("index_type", "metric_type", "state", "total_rows")}
        except Exception as exc:  # noqa: BLE001
            out["dense_index_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["meta"] = self.read_meta()
        except Exception as exc:  # noqa: BLE001
            out["meta_error"] = f"{type(exc).__name__}: {exc}"
        try:
            out["size_mb"] = round(sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file()) / 1e6, 1)
        except OSError:
            pass
        return out

    def count(self, expr: str) -> int:
        """按过滤表达式计数（``count(*)``；Lite 实测支持）。"""
        rows = self.query(filter=expr, output_fields=["count(*)"])
        if rows and isinstance(rows[0], dict):
            for key in ("count(*)", "count"):
                if key in rows[0]:
                    return int(rows[0][key])
        return len(rows or [])

    def health(self) -> dict[str, Any]:
        """``/api/health`` 用的检索状态：任何失败都降级成可读字段，不抛。"""
        if not self.exists():
            return {"ok": False, "reason": "missing", "detail": "data_dir 不存在，需先构建索引", "retrieval_degraded": True}
        try:
            meta = self.read_meta()
        except MilvusUnavailable as exc:
            return {"ok": False, "reason": "unavailable", "detail": str(exc), "retrieval_degraded": True}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "error", "detail": f"{type(exc).__name__}: {exc}", "retrieval_degraded": True}
        missing = self.config.collection not in set(self.client.list_collections())
        return {
            "ok": not missing,
            "reason": "missing_collection" if missing else "ok",
            "collection": self.config.collection,
            "rows": None if missing else self._row_count(),
            "dims": meta.get(META_DIMS),
            "embedding_model": meta.get(META_EMBEDDING_MODEL),
            "built_at": meta.get(META_BUILT_AT),
            "retrieval_degraded": missing,
        }

    def _row_count(self) -> int | None:
        try:
            self._ensure_loaded(self.config.collection)
            return int(self.client.get_collection_stats(self.config.collection).get("row_count", 0))
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------- 启动校验


def validate_retrieval_config(settings: Settings) -> list[str]:
    """按设计 §6 的表做启动校验。

    硬错误（配置自相矛盾）直接抛 ``RuntimeError``，让进程起不来；软问题返回告警列表。
    data_dir 的「被占用」判定由 ``probe_and_release`` 完成（不长期持有）。
    """
    retrieval = settings.retrieval
    embedding: EmbeddingConfig = settings.embedding
    warnings: list[str] = []

    if embedding.model.strip() and not embedding.base_url.strip():
        raise RuntimeError("embedding.model 已配置但 embedding.base_url 为空：请补 base_url 或清空 model（只跑稀疏一路）")
    if not embedding.model.strip():
        warnings.append("embedding.model 未配置：只跑稀疏一路（/api/health 标记 dense_disabled）")

    rerank = settings.rerank
    if rerank.mode == "api" and not (rerank.model.strip() and rerank.base_url.strip()):
        raise RuntimeError("rerank.mode=api 但 rerank.model / rerank.base_url 为空：请补齐配置或改为 mode=auto/off")

    path = retrieval.resolve_milvus_path()
    if not path.is_dir():
        warnings.append(f"Milvus data_dir 不存在（{path}）：检索返回空结果，需先构建索引")
        return warnings

    status = probe_and_release(path)  # 被别的进程占用时抛 MilvusUnavailable
    if status == "ok":
        index = MilvusIndex(config=retrieval, dims=int(embedding.dims), embedding_model=embedding.model)
        meta = index.read_meta()
        index._check_meta(meta)
        index.close()
    return warnings


# ---------------------------------------------------------------- 进程内单例

_INDEX: MilvusIndex | None = None
_INDEX_KEY: tuple[str, int, str] | None = None


def get_milvus_index(settings: Settings | None = None) -> MilvusIndex:
    """进程内单例。配置（路径/dims/模型）变了就换一个句柄。"""
    global _INDEX, _INDEX_KEY
    if settings is None:
        settings = load_settings()
    key = (
        str(settings.retrieval.resolve_milvus_path()),
        int(settings.embedding.dims),
        settings.embedding.model,
    )
    if _INDEX is None or _INDEX_KEY != key:
        if _INDEX is not None:
            _INDEX.close()
        _INDEX = MilvusIndex(
            config=settings.retrieval,
            dims=int(settings.embedding.dims),
            embedding_model=settings.embedding.model,
        )
        _INDEX_KEY = key
    return _INDEX


def reset_milvus_index() -> None:
    """测试用：丢掉单例并释放目录。"""
    global _INDEX, _INDEX_KEY
    if _INDEX is not None:
        _INDEX.close()
    _INDEX = None
    _INDEX_KEY = None


# ---------------------------------------------------------------- CLI


def _render(stats: Mapping[str, Any]) -> str:
    lines = [f"data_dir : {stats.get('path')}", f"collection: {stats.get('collection')}"]
    if not stats.get("exists"):
        lines.append("status   : data_dir 不存在（需先构建索引）")
        return "\n".join(lines)
    lines.append(f"rows     : {stats.get('rows')}（doc {stats.get('doc_rows')} / nav {stats.get('nav_rows')}）")
    index = stats.get("dense_index") or {}
    lines.append(f"dense    : {index.get('index_type')} {index.get('metric_type')} state={index.get('state')}")
    lines.append(f"size     : {stats.get('size_mb')} MB")
    meta = stats.get("meta") or {}
    for key in (META_SCHEMA_VERSION, META_ANALYZER, META_DIMS, META_EMBEDDING_MODEL, META_BUILT_AT, META_CORPUS_FINGERPRINT):
        if key in meta:
            lines.append(f"meta.{key}: {meta[key]}")
    for key in ("rows_error", "doc_rows_error", "nav_rows_error", "dense_index_error", "meta_error"):
        if key in stats:
            lines.append(f"! {key}: {stats[key]}")
    return "\n".join(lines)


def _run_build(settings: Any, *, mode: str, args: Any, json_out: bool) -> int:
    """``--rebuild`` / ``--rebuild-vectors`` / ``--incremental`` 的入口（M4）。"""
    import json

    from .embedding import get_embedding_client
    from .milvus_build import MilvusBuilder

    retrieval = settings.retrieval
    embedding = settings.embedding
    embedder = None if args.no_vectors else get_embedding_client(embedding)
    builder = MilvusBuilder(
        retrieval,
        embedder=embedder,
        wiki_dir=args.wiki_dir or None,
        dims=embedding.dims,
        embedding_model=embedding.model,
    )
    try:
        stats = builder.build(
            mode=mode,
            limit=args.limit,
            vectors=not args.no_vectors,
            progress=None if json_out else print,
        )
    except MilvusUnavailable as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(payload, ensure_ascii=False, indent=1) if json_out else f"错误：{exc}")
        return 1
    finally:
        if embedder is not None:
            embedder.close()
    payload = {"ok": True, **stats.as_dict()}
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    else:
        pass  # progress 已经把每条消息打过了，这里不再重复
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m app.agent.milvus_index", description="Milvus Lite 索引运维")
    parser.add_argument("--stats", action="store_true", help="打印集合统计与元数据")
    parser.add_argument("--health", action="store_true", help="打印 /api/health 口径的检索状态")
    parser.add_argument("--verify", action="store_true", help="校验 data_dir 可打开、集合与 ob_meta 一致")
    parser.add_argument("--rebuild", action="store_true", help="全量重扫重建；未变的块复用向量")
    parser.add_argument("--rebuild-vectors", action="store_true", help="全量重打向量（换模型/dims 后用）")
    parser.add_argument("--incremental", action="store_true", help="只补增量（默认行为）")
    parser.add_argument("--limit", type=int, default=0, help="只构建前 N 个文件（冒烟/自测用）")
    parser.add_argument("--no-vectors", action="store_true", help="只建稀疏索引（CI 无密钥通道）")
    parser.add_argument("--wiki-dir", default="", help="语料目录（默认 backend/doc/ob_wiki）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(list(argv) if argv is not None else None)

    build_mode = ""
    if args.rebuild_vectors:
        build_mode = "rebuild-vectors"
    elif args.rebuild:
        build_mode = "rebuild"
    elif args.incremental:
        build_mode = "incremental"

    settings = load_settings()
    if build_mode:
        return _run_build(settings, mode=build_mode, args=args, json_out=args.json)

    index = get_milvus_index(settings)
    try:
        if args.verify:
            if not index.exists():
                raise MilvusIndexMissing(f"{index.path} 不存在：请先 --rebuild")
            meta = index.ensure_collections(create=False)
            payload: dict[str, Any] = {"ok": True, "meta": meta, "stats": index.stats()}
        elif args.health:
            payload = index.health()
        else:
            payload = index.stats()
    except MilvusUnavailable as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=1))
        else:
            print(f"错误：{exc}")
        return 1
    finally:
        index.close()

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    elif args.health:
        print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    else:
        print(_render(payload.get("stats", payload)))
        if args.verify:
            print(f"verify   : ok（{len(payload.get('meta', {}))} 个 meta 键）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())