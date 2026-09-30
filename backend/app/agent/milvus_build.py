"""Milvus 索引构建 / 增量更新（M4）。

口径见 ``backend/eval/docs/milvus-only-retrieval-design.md`` §7，这里把 7 步落成代码：

1. **扫描** ``backend/doc/ob_wiki``：切分与元数据解析复用 ``doc_index`` 的纯函数
   （``_split_chunks`` / ``_parse_frontmatter`` / ``_detect_mode`` / ``_detect_version``），
   M7 删 FTS5 时这几个函数会搬到检索侧，构建逻辑不用改。
2. **主键** ``pk = xxh64(f"{path}#{section}#{seq}") & 0x7FFF_FFFF_FFFF_FFFF``。``seq`` 是同一
   小节内的块序号——实测 25077 块里有 1630 组 ``(path, section)`` 重复、单组最多 29 块
   （``配置项/集群级别配置项/index.md`` 的 ``文档明细``），不带 ``seq`` 会在 upsert 时静默丢块。
3. **两种文本**（关键设计，别合并）：
   - ``text``（写库、被 BM25 分析）= ``标题×3 + 关键词×2 + 小节 + " | " + 正文``，近似 FTS5
     四列权重，权重可扫（``TITLE_REPEAT`` / ``KEYWORD_REPEAT``）；
   - ``canonical``（送 embedding）= ``标题 + 关键词 + 小节 + " | " + 正文``，**不含重复前缀**。
   于是 ``content_hash = sha256(canonical)[:16] + sha256(text)[:16]`` 分成两段：
   ``canonical`` 段相同 → 向量可复用（M6 调权重不必重打 embedding，只重写 BM25 文本）；
   ``text`` 段也相同 → 整块跳过。行永远由**语料重扫**得出，不从库里的 ``text`` 反推，
   所以 ``--rebuild-vectors`` 能重算出与首次构建完全一致的 embedding 输入。
4. **增量判定**：查回 ``pk → content_hash`` 后分三类（跳过 / 只换文本 / 重打向量）。
5. **剪枝**：库里存在、本轮语料没有的 ``pk`` → ``delete pk in [...]``（分批）。
6. **ob_meta 七键**（六键 + ``text_max_length``）。
7. **原子切换**：一律在 ``<stem>.building-<pid>.db`` 上建（milvus-lite 只认 .db 结尾的路径），成功后目录级 rename 换入，
   旧目录先改名 ``<stem>.old-<pid>.db`` 做回滚点，切换成功再删。

Milvus Lite 单进程：构建时**不能**有别的进程打开同一个 data_dir（构建前有一次
``probe_and_release``，拿不到锁就直接报错提示停服务）。
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
import xxhash
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .doc_index import (
    DEFAULT_DOC_ROOT,
    WIKI_DIRNAME,
    _NAV_FILENAMES,
    _clean_keywords,
    _clean_title,
    _detect_mode,
    _detect_version,
    _parse_frontmatter,
    _split_chunks,
)
from .milvus_index import (
    META_ANALYZER,
    META_BUILT_AT,
    META_CORPUS_FINGERPRINT,
    META_DIMS,
    META_EMBEDDING_MODEL,
    META_SCHEMA_VERSION,
    META_TEXT_MAX_LENGTH,
    SCHEMA_VERSION,
    ChunkRow,
    MilvusIndex,
    MilvusIndexMissing,
    MilvusSchemaMismatch,
    MilvusUnavailable,
    probe_and_release,
)

logger = logging.getLogger(__name__)

#: ``text`` 里标题 / 关键词重复次数（近似 FTS5 的列权重；M6 会扫 ×3/×5、×1/×2）
TITLE_REPEAT = 3
KEYWORD_REPEAT = 2
#: ``content_hash`` 的两段长度（canonical / text），合计 32 字符，刚好等于字段宽度
CANONICAL_HASH_CHARS = 16
TEXT_HASH_CHARS = 16

MODE_INCREMENTAL = "incremental"
MODE_REBUILD = "rebuild"
MODE_REBUILD_VECTORS = "rebuild-vectors"
MODES = (MODE_INCREMENTAL, MODE_REBUILD, MODE_REBUILD_VECTORS)

#: 一轮 embedding + upsert 的块数（控制内存：256 × 1024 float ≈ 2MB）
EMBED_GROUP = 256
#: `pk in [...]` 查询 / 删除的分批大小
QUERY_BATCH = 400
#: 取回旧向量的分批大小（一行 1024 float）
VECTOR_BATCH = 200

#: VARCHAR text 的上限兜底（字节，保守；正常块 ≤ MAX_CHUNK_CHARS 远够用）
DEFAULT_MAX_TEXT_BYTES = 8000

ProgressFn = Callable[[str], None]


# ---------------------------------------------------------------- 纯函数


def chunk_pk(path: str, section: str, seq: int) -> int:
    """块主键：稳定、非负 INT64。同一 (path, section) 的第 ``seq`` 块。"""
    digest = xxhash.xxh64(f"{path}#{section}#{seq}".encode("utf-8")).intdigest()
    return digest & 0x7FFF_FFFF_FFFF_FFFF


def truncate_bytes(value: str, limit: int) -> str:
    """按 UTF-8 字节上限截断（不切坏多字节字符），截断时补 ``…``。

    为什么需要：`_split_chunks` 对**没有 H2/H3 的文件**会走「整篇一块」兜底，
    绕过了 1800 字的切分（实测 6 个文件 8k–40k 字，最大 40363 字），而 Milvus 的
    ``VARCHAR(max_length=8000)`` 会直接拒收，embedding 请求也会超模型输入上限。
    """
    if limit <= 0:
        return value
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    head = raw[: max(0, limit - len("…".encode("utf-8")))].decode("utf-8", errors="ignore")
    return head + "…"


def assemble_text(
    title: str, keywords: str, section: str, body: str, *, max_bytes: int = 0
) -> str:
    """写库的 ``text``：加权前缀 + 正文（BM25 的输入）。"""
    prefix: list[str] = [title] * TITLE_REPEAT + [keywords] * KEYWORD_REPEAT + [section]
    head = " ".join(part for part in prefix if part)
    return truncate_bytes(f"{head} | {body}" if head else body, max_bytes)


def canonical_text(
    title: str, keywords: str, section: str, body: str, *, max_bytes: int = 0
) -> str:
    """送 embedding 的文本：每个字段只出现一次，与权重常量无关。"""
    head = " ".join(part for part in (title, keywords, section) if part)
    return truncate_bytes(f"{head} | {body}" if head else body, max_bytes)


def content_hash(canonical: str, text: str) -> str:
    """前 16 字符是 canonical 段、后 16 字符是 text 段（见模块说明）。"""
    c = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:CANONICAL_HASH_CHARS]
    t = hashlib.sha256(text.encode("utf-8")).hexdigest()[:TEXT_HASH_CHARS]
    return c + t


def canonical_hash_of(value: str) -> str:
    return (value or "")[:CANONICAL_HASH_CHARS]


def corpus_fingerprint(wiki_dir: Path | str) -> str:
    """语料指纹 ``文件数:总字节:最新 mtime_ns``（与 FTS5 侧同口径，不含 TTL 缓存）。"""
    files = size = 0
    newest = 0
    for path in Path(wiki_dir).rglob("*.md"):
        try:
            stat = path.stat()
        except OSError:
            continue
        files += 1
        size += stat.st_size
        newest = max(newest, stat.st_mtime_ns)
    return f"{files}:{size}:{newest}"


# ---------------------------------------------------------------- 数据


@dataclass(frozen=True)
class ChunkPayload:
    """一条待写入的块（由语料扫描得出，与库无关）。"""

    pk: int
    path: str
    section: str
    title: str
    keywords: str
    mode: str
    version: str
    kind: str
    text: str
    canonical: str
    canonical_hash: str
    text_hash: str
    truncated: bool = False  # 该块超过 max_text_bytes，已被截断（H1-only 文件切不出小节）

    @property
    def content_hash(self) -> str:
        return self.canonical_hash + self.text_hash


@dataclass
class BuildStats:
    """一次构建的账本（CLI 的 ``--json`` 直接输出）。"""

    mode: str = MODE_INCREMENTAL
    files: int = 0
    chunks: int = 0
    doc_rows: int = 0
    nav_rows: int = 0
    skipped: int = 0  # 双哈希都一致 → 整块没动
    rewritten: int = 0  # 只换了 text（权重变更），复用旧向量
    embedded: int = 0  # 重新打了向量
    pruned: int = 0
    truncated: int = 0  # 超长块被按 max_text_bytes 截断
    upserted: int = 0
    skipped_build: bool = False  # 指纹未变 + 元数据一致 → 整体跳过
    vectors: bool = True
    fingerprint: str = ""
    data_dir: str = ""
    seconds: float = 0.0
    swapped: bool = False
    verify: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if self.skipped_build:
            return f"语料未变，跳过构建（{self.chunks} 块，指纹 {self.fingerprint}）"
        trunc = f" + 截断 {self.truncated}" if self.truncated else ""
        return (
            f"{self.mode}：{self.files} 文件 / {self.chunks} 块（doc {self.doc_rows} / nav {self.nav_rows}）"
            f"复用 {self.skipped} + 重写 {self.rewritten} + 重打向量 {self.embedded}{trunc}"
            f" + 剪枝 {self.pruned}，{self.seconds:.1f}s"
        )


# ---------------------------------------------------------------- 扫描


def scan_corpus(
    wiki_dir: Path | str, *, limit: int = 0, max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES
) -> list[ChunkPayload]:
    """扫描语料，产出全部块（``limit>0`` 只取前 N 个文件，冒烟用）。

    ``max_text_bytes``：单块 ``text`` / ``canonical`` 的字节上限（VARCHAR 与 embedding 输入双重守卫）。
    """
    wiki = Path(wiki_dir)
    if not wiki.is_dir():
        raise MilvusIndexMissing(
            f"文档库不存在：{wiki}（请先解压 backend/doc/ob_wiki.zip）"
        )
    files = sorted(wiki.rglob("*.md"))
    if limit and limit > 0:
        files = files[: int(limit)]
    payloads: list[ChunkPayload] = []
    by_pk: dict[int, ChunkPayload] = {}
    for path in files:
        rel = path.relative_to(wiki).as_posix()
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # 单篇读失败不该让整库建不起来
            logger.warning("跳过无法读取的文档 %s: %s", rel, exc)
            continue
        meta, body = _parse_frontmatter(raw)
        stem = _clean_title(path.stem)
        disp_title = _clean_title(meta.get("title") or stem)
        title = " ".join(
            part for part in (disp_title, stem, meta.get("description", "")) if part
        )
        keywords = _clean_keywords(meta.get("keywords", ""))
        mode = _detect_mode(rel)
        kind = "nav" if path.name.lower() in _NAV_FILENAMES else "doc"
        seq: dict[str, int] = {}
        for section, chunk in _split_chunks(body):
            index = seq.get(section, 0)
            seq[section] = index + 1
            pk = chunk_pk(rel, section, index)
            chunk = chunk.strip()  # 切出来的块首尾带换行，去掉让 embedding 输入更干净
            raw_text = assemble_text(title, keywords, section, chunk)
            raw_canonical = canonical_text(title, keywords, section, chunk)
            text = truncate_bytes(raw_text, max_text_bytes)
            canonical = truncate_bytes(raw_canonical, max_text_bytes)
            truncated = text != raw_text or canonical != raw_canonical
            digest = content_hash(canonical, text)
            payload = ChunkPayload(
                pk=pk,
                path=rel,
                section=section,
                title=title,
                keywords=keywords,
                mode=mode,
                version=_detect_version(section, rel),
                kind=kind,
                text=text,
                canonical=canonical,
                canonical_hash=digest[:CANONICAL_HASH_CHARS],
                text_hash=digest[CANONICAL_HASH_CHARS:],
                truncated=truncated,
            )
            previous = by_pk.get(pk)
            if previous is not None:
                # 理论上不该发生；真撞了就报出来，别静默丢块
                raise MilvusUnavailable(
                    f"主键冲突：pk={pk} 同时来自 {previous.path!r}/{previous.section!r} "
                    f"与 {rel!r}/{section!r}，请检查切分口径"
                )
            by_pk[pk] = payload
            payloads.append(payload)
    return payloads


# ---------------------------------------------------------------- 构建


@dataclass
class MilvusBuilder:
    """索引构建器。``embedder`` 需要 ``embed_documents(list[str]) -> list[list[float]]``。"""

    config: Any
    embedder: Any = None
    wiki_dir: Path | str | None = None
    dims: int = 1024
    embedding_model: str = ""
    data_dir: Path | str | None = None

    @property
    def wiki(self) -> Path:
        return Path(self.wiki_dir) if self.wiki_dir else DEFAULT_DOC_ROOT / WIKI_DIRNAME

    @property
    def live_dir(self) -> Path:
        if self.data_dir is not None:
            return Path(self.data_dir)
        return self.config.resolve_milvus_path()

    def _index(self, path: Path) -> MilvusIndex:
        return MilvusIndex(
            self.config,
            dims=self.dims,
            embedding_model=self.embedding_model,
            path_override=path,
        )

    # ---- 读旧库 ----

    def _read_existing(
        self, index: MilvusIndex, pks: Sequence[int] | None = None
    ) -> dict[int, str]:
        """查回 ``pk → content_hash``。

        ``pks=None`` 时用 ``query_iterator`` 扫**全库**：增量剪枝必须知道旧库里出现过、
        但本轮语料已经没有了的 pk，只查本轮 pk 是查不到待删块的（M4 测试踩到）。
        """
        found: dict[int, str] = {}
        if pks is None:
            iterator = index.query_iterator(
                batch_size=QUERY_BATCH, filter="pk >= 0", output_fields=["pk", "content_hash"]
            )
            try:
                while True:
                    rows = iterator.next()
                    if not rows:
                        break
                    for row in rows:
                        found[int(row["pk"])] = str(row.get("content_hash") or "")
            finally:
                iterator.close()
            return found
        for batch in _batches(pks, QUERY_BATCH):
            expr = "pk in [" + ",".join(str(pk) for pk in batch) + "]"
            rows = index.query(filter=expr, output_fields=["pk", "content_hash"], limit=len(batch))
            for row in rows:
                found[int(row["pk"])] = str(row.get("content_hash") or "")
        return found

    def _read_vectors(self, index: MilvusIndex, pks: Sequence[int]) -> dict[int, list[float]]:
        """取回旧向量（只换文本块的复用路径）。"""
        found: dict[int, list[float]] = {}
        for batch in _batches(pks, VECTOR_BATCH):
            expr = "pk in [" + ",".join(str(pk) for pk in batch) + "]"
            rows = index.query(filter=expr, output_fields=["pk", "vector"], limit=len(batch))
            for row in rows:
                vector = row.get("vector")
                if vector is not None:
                    found[int(row["pk"])] = list(vector)
        return found

    # ---- 主流程 ----

    def build(
        self,
        *,
        mode: str = MODE_INCREMENTAL,
        limit: int = 0,
        vectors: bool = True,
        progress: ProgressFn | None = None,
    ) -> BuildStats:
        if mode not in MODES:
            raise ValueError(f"未知构建模式 {mode!r}，可选 {MODES}")
        started = time.monotonic()
        stats = BuildStats(mode=mode, vectors=vectors)

        def say(message: str) -> None:
            logger.info("%s", message)
            if progress is not None:
                progress(message)

        wiki = self.wiki
        payloads = scan_corpus(wiki, limit=limit, max_text_bytes=self.config.max_text_bytes)
        # 构建指纹 = 语料指纹 + 加权口径：改标题/关键词前缀权重时必须让增量短路失效，
        # 这样才会走「只重写 text、复用旧向量」的廉价路径（M6 扫权重要用）。
        fingerprint = corpus_fingerprint(wiki) + f":t{TITLE_REPEAT}k{KEYWORD_REPEAT}"
        stats.files = len({p.path for p in payloads})
        stats.chunks = len(payloads)
        stats.doc_rows = sum(1 for p in payloads if p.kind == "doc")
        stats.nav_rows = stats.chunks - stats.doc_rows
        stats.fingerprint = fingerprint
        stats.truncated = sum(1 for p in payloads if p.truncated)
        live = self.live_dir
        stats.data_dir = str(live)
        say(f"扫描完成：{stats.files} 文件 / {stats.chunks} 块（doc {stats.doc_rows} / nav {stats.nav_rows}）")

        if vectors and self.embedder is None:
            raise MilvusUnavailable(
                "未配置 embedding 客户端：无法构建稠密向量"
                "（如需无密钥构建稀疏索引，请加 --no-vectors）"
            )

        # 指纹短路：语料没变 + 元数据口径一致 → 整库跳过（省掉全量哈希比对）
        live_meta = self._live_meta(live) if live.is_dir() else None
        model_changed = bool(
            vectors
            and live_meta is not None
            and self.embedding_model
            and str(live_meta.get(META_EMBEDDING_MODEL) or "") != self.embedding_model
        )
        if (
            mode == MODE_INCREMENTAL
            and live_meta is not None
            and not model_changed
            and live_meta.get(META_CORPUS_FINGERPRINT) == fingerprint
        ):
            stats.skipped_build = True
            stats.seconds = time.monotonic() - started
            say("语料指纹未变，跳过构建")
            return stats
        if model_changed:
            say(
                f"embedding 模型变了（{live_meta.get(META_EMBEDDING_MODEL)!r} → "
                f"{self.embedding_model!r}），本轮全量重打向量"
            )

        tmp = _side_dir(live, "building")
        self._prepare_tmp(tmp, live, fresh=mode == MODE_REBUILD_VECTORS)
        index = self._index(tmp)
        try:
            index.ensure_collections(create=True, rows=len(payloads) or None)
            reembed = mode == MODE_REBUILD_VECTORS or model_changed
            existing: dict[int, str] = {}
            if not reembed:
                existing = self._read_existing(index)

            to_embed: list[ChunkPayload] = []
            to_rewrite: list[ChunkPayload] = []
            for payload in payloads:
                old = None if reembed else existing.get(payload.pk)
                if old is None:
                    to_embed.append(payload)
                elif old == payload.content_hash:
                    stats.skipped += 1
                elif canonical_hash_of(old) == payload.canonical_hash:
                    to_rewrite.append(payload)  # 权重变了，向量还能用
                else:
                    to_embed.append(payload)
            say(
                f"增量判定：复用 {stats.skipped} / 重写 {len(to_rewrite)} / 重打向量 {len(to_embed)}"
            )

            zero = [0.0] * int(self.dims)
            if to_rewrite:
                old_vectors = self._read_vectors(index, [p.pk for p in to_rewrite])
                rewritten: list[ChunkPayload] = []
                for payload in to_rewrite:
                    vector = old_vectors.get(payload.pk)
                    if vector is None:  # 旧向量读不到就退回重打，不能写零向量骗检索
                        to_embed.append(payload)
                    else:
                        rewritten.append(payload)
                        self._upsert_rows(index, [(payload, vector)], zero=zero)
                stats.rewritten = len(rewritten)
                stats.upserted += len(rewritten)

            if to_embed:
                stats.embedded, stats.upserted = self._embed_and_upsert(
                    index, to_embed, vectors=vectors, zero=zero, say=say, upserted=stats.upserted
                )
            elif not vectors:
                stats.embedded = 0

            stale = [pk for pk in existing if pk not in {p.pk for p in payloads}]
            for batch in _batches(stale, QUERY_BATCH):
                index.delete("pk in [" + ",".join(str(pk) for pk in batch) + "]")
                stats.pruned += len(batch)
            if stale:
                say(f"剪枝：删除 {stats.pruned} 个过期块")

            index.write_meta(
                {
                    META_SCHEMA_VERSION: SCHEMA_VERSION,
                    META_CORPUS_FINGERPRINT: fingerprint,
                    META_EMBEDDING_MODEL: self.embedding_model if vectors else "",
                    META_DIMS: int(self.dims),
                    META_ANALYZER: self.config.analyzer,
                    META_BUILT_AT: time.strftime("%Y-%m-%d %H:%M:%S"),
                    META_TEXT_MAX_LENGTH: int(self.config.max_text_bytes),
                }
            )
        except Exception:
            index.close()
            shutil.rmtree(tmp, ignore_errors=True)  # 失败不留半个索引
            raise
        finally:
            index.close()  # close + release_server(tmp)，之后才能 rename

        self._swap(live, tmp)
        stats.swapped = True
        stats.verify = probe_and_release(live)
        stats.seconds = time.monotonic() - started
        say(f"构建完成：{stats.data_dir}（{stats.verify}），{stats.seconds:.1f}s")
        return stats

    # ---- 步骤实现 ----

    def _live_meta(self, live: Path) -> dict[str, str] | None:
        """读线上库的 ob_meta；结构不一致时返回 None（交给后续重建报错）。"""
        probe = self._index(live)
        try:
            return probe.ensure_collections(create=False)
        except MilvusSchemaMismatch:
            return None
        finally:
            probe.close()

    def _prepare_tmp(self, tmp: Path, live: Path, *, fresh: bool) -> None:
        if tmp.exists():
            shutil.rmtree(tmp)
        if fresh or not live.is_dir():
            tmp.mkdir(parents=True)
            return
        # 先确认没有别的进程占着 data_dir（Milvus Lite 单进程），再整目录复制
        try:
            probe_and_release(live)
        except MilvusUnavailable as exc:
            raise MilvusUnavailable(
                f"{live} 正被其它进程占用：请先停掉服务（uvicorn 需 --workers 1）再构建。{exc}"
            ) from exc
        shutil.copytree(live, tmp)

    def _embed_and_upsert(
        self,
        index: MilvusIndex,
        payloads: Sequence[ChunkPayload],
        *,
        vectors: bool,
        zero: list[float],
        say: ProgressFn,
        upserted: int,
    ) -> tuple[int, int]:
        embedded = 0
        for batch in _batches(payloads, EMBED_GROUP):
            docs = [p for p in batch if p.kind != "nav"]
            navs = [p for p in batch if p.kind == "nav"]
            rows: list[tuple[ChunkPayload, Sequence[float] | None]] = []
            if docs and vectors:
                texts = [p.canonical for p in docs]
                vectors_list = self.embedder.embed_documents(texts)
                if len(vectors_list) != len(texts):
                    raise MilvusUnavailable(
                        f"embedding 返回条数不符：期望 {len(texts)}，实得 {len(vectors_list)}"
                    )
                rows.extend(zip(docs, vectors_list))
                embedded += len(docs)
            else:
                rows.extend((p, zero) for p in docs)  # --no-vectors：只建稀疏索引
            # 导航页恒零向量（稠密一路会 filter kind == "doc"），不花 embedding 配额
            rows.extend((p, zero) for p in navs)
            self._upsert_rows(index, rows, zero=zero)
            upserted += len(rows)
            say(f"向量化 {embedded}/{len(payloads)}")
        return embedded, upserted

    def _upsert_rows(
        self,
        index: MilvusIndex,
        rows: Iterable[tuple[ChunkPayload, Sequence[float] | None]],
        *,
        zero: list[float],
    ) -> None:
        entities = []
        for payload, vector in rows:
            row = ChunkRow(
                pk=payload.pk,
                text=payload.text,
                kind=payload.kind,
                path=payload.path,
                section=payload.section,
                title=payload.title,
                mode=payload.mode,
                version=payload.version,
                content_hash=payload.content_hash,
                vector=list(vector) if vector is not None else None,
            )
            entities.append(row.to_entity(zero_vector=zero))
        index.upsert(entities)

    def _swap(self, live: Path, tmp: Path) -> None:
        """目录级原子切换：live → live.old-<pid>，tmp → live，成功后再删 .old。"""
        backup = _side_dir(live, "old")
        if backup.exists():
            shutil.rmtree(backup)
        if live.exists():
            live.rename(backup)
        try:
            tmp.rename(live)
        except Exception:
            if backup.exists() and not live.exists():
                backup.rename(live)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _side_dir(live: Path, tag: str) -> Path:
    """构建/回滚用的旁路目录。

    **必须以 .db 结尾**：milvus-lite 只认这种本地路径（``ob_wiki.milvus`` 会被直接拒绝），
    所以不能简单地在名字后面接后缀。
    """
    stem = live.stem if live.suffix == ".db" else live.name
    return live.with_name(f"{stem}.{tag}-{os.getpid()}.db")


def _batches(items: Sequence[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]