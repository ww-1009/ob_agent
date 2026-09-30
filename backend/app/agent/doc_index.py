"""本地知识库检索：ob_wiki 官方文档的检索门面与精读读取。

为什么需要它：文档库有 5000+ 篇 Markdown（22 MB 正文），而文件工具只有
``list_directory`` / ``read_file``。模型只能靠目录名逐级猜、再把整篇文档读进上下文，
既慢又容易被大文档挤爆上下文，且中文关键词（事务/索引/大表/锁）与目录名常常对不上。
所以先 ``search`` 命中到「文档 + 小节」，再用 ``read`` 只精读该小节。

检索本身在 ``app.agent.retrieval``：Milvus Lite 单集合里放 BM25 稀疏索引与 1024 维稠密
向量，三路（sparse / dense / hybrid）各自召回后在客户端融合 + 列权重重排。本模块留下的是
**与索引引擎无关的那部分**：切分口径（``_split_chunks``）、元数据解析、模式/版本识别、
导航小节判定、摘要、默认引擎选择，以及 ``read`` 的文件读取。

M7（2026-09）删掉了 SQLite FTS5 一路：索引不再是本模块的职责，而是派生数据，由
``python -m app.agent.milvus_index --rebuild`` 构建（设计见
``backend/eval/docs/milvus-only-retrieval-design.md``）。检索层任何意外都降级成空结果 +
warning，不会下发成 500。
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# 默认位置：与 FileManagementToolkit 的 root_dir 一致（后端以 backend/ 为工作目录）
DEFAULT_DOC_ROOT = Path("./doc")
WIKI_DIRNAME = "ob_wiki"

# 单个小节块的最大字符数：超长文档按空行再切，避免一个块覆盖整篇
MAX_CHUNK_CHARS = 1800
# read_doc 默认/上限返回字符数：整篇最大 270 KB，不设上限会直接挤爆上下文
DEFAULT_READ_CHARS = 6000
MAX_READ_CHARS = 20000
DEFAULT_LIMIT = 5
MAX_LIMIT = 20
# 同一次检索里同一篇文档最多返回几个小节，避免整屏都是同一篇（如"大表创建索引"）
MAX_CHUNKS_PER_PATH = 2
#: 默认检索引擎（M7 删 FTS5 后 = Milvus 稀疏）；hybrid 过门禁后再改配置默认值
DEFAULT_RETRIEVER = "sparse"

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_H2_H3 = re.compile(r"^(#{2,3})\s+(.*)$")
_H1 = re.compile(r"^#\s+(.*)$")
_VERSION = re.compile(r"[Vv](\d+\.\d+(?:\.\d+)*)")
# 只指路不讲答案的小节名（按「父 > 子」里的最后一段判断，见 _is_navigation_section）
_NAV_SECTION = re.compile(r"相关(文档|阅读|链接|内容|问题)|参考(文档|资料|信息)|更多(信息|内容)|参见|延伸阅读|附录")
_MIN_CHUNK_SPLIT = 200
# 导航型文件：分类索引 + 知识库检索指南，按文件名判定（实测按链接密度判定会误伤正文：
# 根 README.md 密度仅 0.08 却是说明文，而 V4.2.5 文档更新记录.md 这类真正文密度高达 0.88）。
_NAV_FILENAMES = frozenset({"index.md", "readme.md"})
# 导入残留：文件名统一后缀 -OceanBase；keywords 里混入站点标签
_FILENAME_SUFFIX = "-OceanBase"
_KEYWORD_NOISE = frozenset({"OB Cloud 云数据库", "OB Cloud", "OceanBase", "OceanBase 数据库"})
_MODE_PATTERNS = (
    ("Oracle", re.compile(r"（Oracle 模式）|Oracle 租户|Oracle 模式", re.I)),
    ("MySQL", re.compile(r"（MySQL 模式）|MySQL 租户|MySQL 模式", re.I)),
    ("sys", re.compile(r"sys 租户|系统租户", re.I)),
)
# 查询改写用的同义词（键只要作为子串出现在提问里就展开）。
# 只收「用户口语 ↔ 文档用词」这种真正会漏召的对照，不做泛化，避免把排序冲散。
_SYNONYMS: dict[str, tuple[str, ...]] = {
    "水位": ("资源使用率", "已分配", "使用率"),
    "合并": ("major freeze", "major_freeze", "minor freeze", "转储"),
    "转储": ("minor freeze", "memstore"),
    "大表": ("胖表", "大表建索引"),
    "慢查询": ("慢 SQL", "slow query", "慢sql"),
    "死锁": ("deadlock", "锁等待"),
    "锁等待": ("lock wait", "行锁", "锁冲突"),
    "超时": ("timeout", "ob_query_timeout", "超时时间"),
    "执行计划": ("explain", "计划缓存", "plan cache"),
    "隔离级别": ("isolation level", "读已提交", "可重复读"),
    "副本": ("replica", "locality", "多副本"),
    "分区": ("partition", "分区表", "分区裁剪"),
    "备份": ("backup", "备份恢复"),
    "恢复": ("restore", "备份恢复"),
    "扩容": ("scale out", "扩容节点", "资源扩容"),
    "缩容": ("scale in",),
    "参数": ("配置项", "系统变量"),
    "报错": ("错误码", "error code"),
    "升级": ("upgrade", "版本升级"),
    "迁移": ("migration", "数据迁移"),
    "统计信息": ("optimizer statistics", "gather statistics", "收集统计信息"),
    "索引": ("index", "局部索引", "全局索引"),
    "事务": ("transaction", "trans_id"),
    "主键": ("primary key",),
    "资源隔离": ("resource manager", "resource unit", "cgroup"),
    "限流": ("throttle", "流控"),
    "闪回": ("flashback",),
}


class DocIndexError(RuntimeError):
    """检索层可预期错误（由工具层转成 ok:false，不下发堆栈）。"""


class DocPathError(DocIndexError):
    """请求的文档路径越界或不存在。"""


def _clean_title(name: str) -> str:
    return name[:-len(_FILENAME_SUFFIX)] if name.endswith(_FILENAME_SUFFIX) else name


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip().lower()] = value.strip()
    return meta, text[match.end():]


def _clean_keywords(raw: str) -> str:
    words = [w.strip() for w in re.split(r"[,，]", raw) if w.strip()]
    return " ".join(w for w in words if w not in _KEYWORD_NOISE)


def _split_chunks(body: str) -> list[tuple[str, str]]:
    """按 H2/H3 切片，保留小节路径（``父 > 子``）；无小节标题的文档退化成整篇。

    超长块（> ``MAX_CHUNK_CHARS``）按空行优先再切；**没有 H2/H3 的兜底路径也必须切**——
    M4 实测真语料里有 6 篇这样的长文件（最大 ``obshell/错误码.md`` 40363 字），
    不切就只能按 ``max_text_bytes`` 截断，等于把后半篇内容丢出索引（M7 根治）。
    """
    chunks: list[tuple[str, str]] = []
    trail: dict[int, str] = {}
    current: str | None = None
    buf: list[str] = []
    h1 = ""

    def hard_split(text: str) -> list[str]:
        """按空行边界切成 <= MAX_CHUNK_CHARS 的若干块（找不到空行就硬切）。"""
        parts: list[str] = []
        while len(text) > MAX_CHUNK_CHARS:
            at = text.rfind("\n\n", 0, MAX_CHUNK_CHARS)
            at = at if at > _MIN_CHUNK_SPLIT else MAX_CHUNK_CHARS
            parts.append(text[:at])
            text = text[at:]
        if text.strip():
            parts.append(text)
        return parts

    def flush() -> None:
        if not buf or current is None:
            return
        chunks.extend((current, part) for part in hard_split("\n".join(buf)))

    for line in body.splitlines():
        heading = _H2_H3.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            trail = {k: v for k, v in trail.items() if k < level}
            trail[level] = heading.group(2).strip()
            current = " > ".join(trail[k] for k in sorted(trail))
            buf = []
            continue
        if not h1:
            top = _H1.match(line)
            if top:
                h1 = top.group(1).strip()
        buf.append(line)
    flush()
    if not chunks:
        # 整篇只有 H1（或完全没有标题）：小块也统一用 H1 当小节名，read 的 section 过滤才可用
        chunks = [(h1 or "正文", part) for part in (hard_split(body) or [body])]
    return chunks


def _detect_mode(rel_path: str) -> str:
    for name, pattern in _MODE_PATTERNS:
        if pattern.search(rel_path):
            return name
    return ""


# 注意不能用 str.capitalize()："MySQL".capitalize() == "Mysql"，会把模式过滤悄悄变成永假条件
_MODE_CANON = {"mysql": "MySQL", "oracle": "Oracle", "sys": "sys"}


def _normalize_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    return _MODE_CANON.get(key, (mode or "").strip())


def _detect_version(section: str, rel_path: str) -> str:
    match = _VERSION.search(section) or _VERSION.search(rel_path)
    return match.group(1) if match else ""


class DocIndex:
    """ob_wiki 文档库的检索与精读门面（索引由 ``app.agent.milvus_index`` 构建）。"""

    def __init__(
        self,
        doc_root: Path | str = DEFAULT_DOC_ROOT,
        *,
        wiki_dirname: str = WIKI_DIRNAME,
    ) -> None:
        self.doc_root = Path(doc_root)
        self.wiki_dirname = wiki_dirname
        self.wiki_dir = self.doc_root / wiki_dirname
        # 最近一次 Milvus 检索的账本（degraded/pool/耗时），M8 报告与排查用
        self.last_retrieval: Any = None

    # ---- 路径 ----

    def to_doc_path(self, wiki_rel: str) -> str:
        """库内相对路径 → 文件工具能用的路径（``ob_wiki/xxx.md``）。"""
        return f"{self.wiki_dirname}/{wiki_rel}"

    def to_wiki_path(self, user_path: str) -> str:
        """把模型给的路径收敛成库内相对路径，越界一律拒绝。"""
        raw = (user_path or "").strip().replace("\\", "/")
        if not raw:
            raise DocPathError("path 不能为空")
        if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
            raise DocPathError("只接受相对 doc 目录的路径，不允许绝对路径")
        parts = [p for p in PurePosixPath(raw).parts if p not in (".", "")]
        if any(p == ".." for p in parts):
            raise DocPathError("路径不允许包含 ..")
        # 容忍模型多带一层 doc/ 或 ob_wiki/（prompt 里两者都出现过）
        if parts and parts[0] == "doc":
            parts = parts[1:]
        if parts and parts[0] == self.wiki_dirname:
            parts = parts[1:]
        if not parts:
            raise DocPathError("路径为空或只指向文档库根目录")
        wiki_rel = "/".join(parts)
        if not self._resolve(wiki_rel).is_file():
            raise DocPathError(f"文档不存在：{self.to_doc_path(wiki_rel)}")
        return wiki_rel

    def _resolve(self, wiki_rel: str) -> Path:
        path = (self.wiki_dir / wiki_rel).resolve()
        root = self.wiki_dir.resolve()
        if not path.is_relative_to(root):
            raise DocPathError("路径越出文档库范围")
        return path

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        mode: str = "",
        version: str = "",
        include_index: bool = False,
        retriever: str = "",
    ) -> list[dict[str, Any]]:
        """检索文档小节（Milvus 三路，见 ``app.agent.retrieval``）。

        ``include_index=True`` 让导航页（``index.md`` / ``README.md``）正常参与，适合
        「有哪些 / 包含哪些 / 怎么分类」这类要清单的提问；默认把它们排除在召回之外
        （导航页抢 top1 是硬门禁，见设计文档 §8.2）。

        ``retriever`` 选引擎：``""``（默认）用配置 ``retrieval.default_retriever``，
        或显式 ``sparse`` / ``dense`` / ``hybrid``。
        """
        if not query or not query.strip():
            raise DocIndexError("query 不能为空")
        engine = (retriever or _default_retriever()).strip().lower()
        limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
        # 提问里自带模式/版本时自动消歧（文档库有 MySQL/Oracle 双份同名文档，混着答会答错）。
        # 只当提问**明确写了**才过滤，且过滤后若一条不剩就退回不过滤的检索。
        auto_mode = "" if mode else _detect_mode(query)
        found = None if version else _VERSION.search(query)
        auto_version = found.group(1) if found else ""
        return self._search_milvus(
            query,
            engine,
            limit=limit,
            mode=mode or auto_mode,
            version=version or auto_version,
            auto_mode=auto_mode,
            auto_version=auto_version,
            include_index=include_index,
        )

    def _search_milvus(
        self,
        query: str,
        engine: str,
        *,
        limit: int,
        mode: str,
        version: str,
        auto_mode: str,
        auto_version: str,
        include_index: bool,
    ) -> list[dict[str, Any]]:
        """Milvus 三路检索：降级只记日志、返回已有结果，任何意外都不下发成 500。"""
        from app.agent import retrieval as retrieval_module

        self.last_retrieval = None
        try:
            retriever = retrieval_module.get_retriever()
            result = retriever.search(
                query,
                limit=limit,
                mode=mode,
                version=version,
                include_index=include_index,
                retriever=engine,
            )
            if not result.entries and (auto_mode or auto_version):
                # 自动消歧后一条不剩就退回不过滤，与 search() 的注释同口径
                retry = retriever.search(
                    query,
                    limit=limit,
                    include_index=include_index,
                    retriever=engine,
                )
                if retry.entries:
                    result = retry
        except Exception as exc:  # noqa: BLE001 - 检索层意外不该变成 500
            logger.warning("Milvus 检索失败（retriever=%s）：%s", engine, exc)
            return []
        self.last_retrieval = result
        if result.degraded:
            logger.warning(
                "检索降级（retriever=%s）：%s", engine, {k: v for k, v in result.as_dict().items() if k != "entries"}
            )
        return result.entries

    def read(
        self,
        user_path: str,
        *,
        section: str = "",
        max_chars: int = DEFAULT_READ_CHARS,
    ) -> dict[str, Any]:
        """精读一篇文档（或其中一个小节）。

        直接从 wiki 文件读、用与建索引**同一套** ``_split_chunks`` 切片：M7 起 read 不再依赖
        任何索引库（索引是派生数据、可能还没建或正在重建），而且能看到整篇文档的全部小节
        （FTS5 时代 ``chunks`` 表每篇最多只有 ``MAX_CHUNKS_PER_PATH`` 块，读出来是残缺的）。
        """
        wiki_rel = self.to_wiki_path(user_path)
        max_chars = max(200, min(int(max_chars or DEFAULT_READ_CHARS), MAX_READ_CHARS))
        raw = self._resolve(wiki_rel).read_text(encoding="utf-8", errors="replace")
        meta, body = _parse_frontmatter(raw)
        title = (meta.get("title") or "").strip() or _clean_title(Path(wiki_rel).stem)
        chunks = _split_chunks(body)
        sections = [{"section": name, "chars": len(text)} for name, text in chunks]
        total = sum(len(text) for _, text in chunks)
        if section:
            picked = [text for name, text in chunks if section.strip().lower() in (name or "").lower()]
            if not picked:
                raise DocPathError(
                    f"该文档没有匹配的小节：{section}；可用小节见 "
                    f"{self.to_doc_path(wiki_rel)} 的 sections 列表"
                )
        else:
            picked = [text for _, text in chunks]
        text = "\n\n".join(picked)
        return {
            "path": self.to_doc_path(wiki_rel),
            "title": title,
            "section": section.strip(),
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
            "total_chars": total,
            # 小节清单兼任目录：模型据此决定下一步读哪一节，而不是把整篇拉进上下文
            "sections": sections if (not section or len(sections) > 1) else [],
        }


def _is_navigation_section(section: str) -> bool:
    return bool(_NAV_SECTION.search((section or "").rsplit(" > ", 1)[-1].strip()))


def _excerpt(body: str, tokens: Sequence[str], width: int = 220) -> str:
    """正文里以首个命中词为中心的摘要。

    token 由 ``retrieval._column_tokens`` 给（jieba 词），落在标题/小节名上的命中在正文里
    找不到，就地退化成正文开头一段。
    """
    text = re.sub(r"\s+", " ", body or "").strip()
    if not text:
        return ""
    pos = -1
    for token in tokens:
        idx = text.find(token)
        if idx >= 0 and (pos < 0 or idx < pos):
            pos = idx
    if pos < 0:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    return f"{'…' if start > 0 else ''}{text[start:end]}{'…' if end < len(text) else ''}"


def _finalize(entries: list[dict[str, Any]], limit: int, *, prefix: str = WIKI_DIRNAME) -> list[dict[str, Any]]:
    """同一篇文档最多留 MAX_CHUNKS_PER_PATH 个小节，并把库内路径补成文件工具能用的路径。"""
    out: list[dict[str, Any]] = []
    per_path: dict[str, int] = {}
    for entry in entries:
        path = entry["wiki_path"]
        if per_path.get(path, 0) >= MAX_CHUNKS_PER_PATH:
            continue
        per_path[path] = per_path.get(path, 0) + 1
        item = {k: v for k, v in entry.items() if k != "wiki_path"}
        item["path"] = f"{prefix}/{path}"
        out.append(item)
        if len(out) >= limit:
            break
    return out


# ---- 模块级单例（工具层用）----

_default_lock = threading.Lock()
_default: DocIndex | None = None

#: ``_default_retriever()`` 的结果按进程缓存：``load_settings()`` 实测 ~5.8 ms/次
#: （重读 .env + 解析 config.yaml），放在每次检索的路径上等于给每问加 5.8 ms。
_default_retriever_cache: str | None = None


def _default_retriever() -> str:
    """配置 ``retrieval.default_retriever``；读不到就退回 ``DEFAULT_RETRIEVER``（sparse）。

    进程内只读一次：引擎选择在启动后不该漂移（改配置 = 重启，与其它配置项同口径）。
    """
    global _default_retriever_cache
    if _default_retriever_cache is None:
        try:
            from app.config import load_settings

            value = str(load_settings().retrieval.default_retriever or DEFAULT_RETRIEVER)
        except Exception:  # 配置缺失/非法都不该让检索起不来
            value = DEFAULT_RETRIEVER
        _default_retriever_cache = value
    return _default_retriever_cache


def configure(
    doc_root: Path | str | None = None,
    *,
    wiki_dirname: str = WIKI_DIRNAME,
) -> DocIndex:
    """重置默认实例（测试与脚本用）。"""
    global _default
    with _default_lock:
        _default = DocIndex(doc_root or DEFAULT_DOC_ROOT, wiki_dirname=wiki_dirname)
        return _default


def get_index() -> DocIndex:
    global _default
    with _default_lock:
        if _default is None:
            _default = DocIndex()
        return _default
