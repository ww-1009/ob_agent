"""本地知识库检索：把 backend/doc/ob_wiki 的官方文档建成离线全文索引。

为什么需要它：文档库有 5000+ 篇 Markdown（22 MB 正文），而文件工具只有
``list_directory`` / ``read_file``。模型只能靠目录名逐级猜、再把整篇文档读进上下文，
既慢又容易被大文档挤爆上下文，且中文关键词（事务/索引/大表/锁）与目录名常常对不上。
这里把每一篇文档按 H2/H3 切成小节建 SQLite FTS5 索引，用一个 ``search`` 命中到
「文档 + 小节」，再用 ``read`` 只读该小节。

中文分词：SQLite 的 ``trigram`` 分词器只索引 3 字窗口，**2 字词根本匹配不到**
（实测 ``"事务"``/``"索引"``/``"大表"`` 全部 0 命中）。因此这里在入库前把 CJK 串预先切成
「单字 + 双字」token（``事务隔离`` → ``事 务 隔 离 事务 务隔 隔离``），用 unicode61 分词器索引；
查询侧走同一套切分，于是 1~2 字的中文关键词也能命中，并且仍有正常的 bm25 相关性打分。
索引表用 **external content** 挂到存原文的 ``chunks`` 表，所以 ``snippet()`` 返回的仍是
可读的中文原文，而不是切分后的 token。

索引是**派生数据**：不进 git（``/backend/doc/*`` 已忽略），缺失或语料变化时自动重建
（全库实测约 5 秒），也可以显式预热：``python -m app.agent.doc_index --rebuild``。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# 默认位置：与 FileManagementToolkit 的 root_dir 一致（后端以 backend/ 为工作目录）
DEFAULT_DOC_ROOT = Path("./doc")
WIKI_DIRNAME = "ob_wiki"
INDEX_FILENAME = "ob_wiki.index.db"
# 索引结构版本：分块/字段口径变了就 +1，旧索引会被判为过期并自动重建
SCHEMA_VERSION = 2
# 版本命中奖励：提问带了版本号时，标注该版本的小节优先（见 _entry）
VERSION_MATCH_BONUS = 8.0
# 全词命中奖励：一次召回同时命中提问里所有 token 的小节，优先于只命中部分词的小节
AND_ALL_TERMS_BONUS = 10.0
# 同义词召回惩罚：只有原词一条都排不上时，同义词结果才该顶上来
SYNONYM_PENALTY = 10.0
# 导航型小节惩罚：正文里的「相关文档 / 参见 / 更多信息」和 index.md 一样只指路不讲答案，
# 而且往往又短又抄满邻居标题，bm25 天然占优（实测真语料里「慢SQL 诊断」第一篇就是它）。
NAVIGATION_SECTION_PENALTY = 12.0
# 导航型文件惩罚：index.md（分类索引）和根 README.md（知识库检索指南）都只指路不讲答案。
# 但不能像正文那样硬排除 —— 实测「OceanBase 数据库包含哪些分类」过滤疑问词后 README.md 排 #1，
# 它就是该问题的最佳答案。所以默认参与检索但重降权，include_index=True 时才按正常排序。
# 取 40 而不是 12：导航页又短又堆满关键词，bm25 动辄 90 分以上（实测「Oracle 模式的事务隔离级别」
# 里上级 index.md 原始分 91.18，扣 12 后仍排第 3，会插进正文之间）。
NAVIGATION_FILE_PENALTY = 40.0

# 单个小节块的最大字符数：超长文档按空行再切，避免一个块覆盖整篇
MAX_CHUNK_CHARS = 1800
# read_doc 默认/上限返回字符数：整篇最大 270 KB，不设上限会直接挤爆上下文
DEFAULT_READ_CHARS = 6000
MAX_READ_CHARS = 20000
DEFAULT_LIMIT = 5
MAX_LIMIT = 20
# 同一次检索里同一篇文档最多返回几个小节，避免整屏都是同一篇（如"大表创建索引"）
MAX_CHUNKS_PER_PATH = 2
# bm25 列权重：标题 > 关键词 > 小节名 > 正文
_BM25_WEIGHTS = (10.0, 6.0, 4.0, 1.0)

_CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")
_ASCII_WORD = re.compile(r"[0-9a-z_]+")
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
# 中文疑问词与高频虚词，查询侧直接丢掉：清单类提问（「错误码一共有哪些」「系统变量有哪些」）
# 的字面会被 FAQ 类文档里的「哪些/有哪/共有」大量命中，bm25 反而把 FAQ 顶到前三、答案文档
# 一个不进（实测真语料，见 tests/test_doc_index.py 的清单提问用例）。
# 注意：实测「只惩罚单字 token」无效 —— 噪声出在双字上，必须在切分阶段去掉。
_STOPWORDS = frozenset({
    "哪些", "什么", "怎么", "如何", "是否", "一共", "共有", "包含", "包括", "有哪",
    "介绍", "说明", "请问", "可以", "需要", "为什", "多少", "几个", "哪几", "这个",
    "那个", "以及", "或者", "还是", "有没", "没有", "告诉", "帮我", "一下",
    "的", "了", "吗", "呢", "有", "是", "和", "与", "在", "里", "中", "都",
    # 单字形态也要收：中文切分是「双字 + 单字」都出，只收双字的话 FAQ 仍会被 哪/些/一/共 命中
    "哪", "些", "什", "么", "一", "共", "几", "多", "少", "我", "你", "请", "帮",
    "告", "诉", "能", "会", "要", "可", "以",
})
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


class DocIndexMissing(DocIndexError):
    """文档库或索引不可用。"""


class DocPathError(DocIndexError):
    """请求的文档路径越界或不存在。"""


def _cjk_tokens(text: str) -> str:
    """CJK 串切「单字 + 双字」，其余按 ASCII 词切；结果供 unicode61 分词器索引。"""
    out: list[str] = []

    def _run(match: re.Match[str]) -> str:
        run = match.group(0)
        out.extend(run)
        out.extend(run[i:i + 2] for i in range(len(run) - 1))
        return " "

    rest = _CJK_RUN.sub(_run, text.lower())
    out.extend(_ASCII_WORD.findall(rest))
    return " ".join(out)


def _is_cjk(token: str) -> bool:
    return bool(token) and bool(_CJK_RUN.fullmatch(token))


def _query_tokens(query: str) -> list[str]:
    """查询侧切分 + 去掉噪声 token。

    单个 ASCII 字符/数字（``V4.2.5`` 里的 ``2``、``5``）区分度极低，会把 AND 收紧到 0 命中，
    直接丢掉；单个汉字保留 —— 它是有意义的词（``慢SQL`` 的 ``慢``、``锁``），而且长中文串的
    字符已经被双字 token 覆盖，多留一个单字不会明显收紧召回。

    再去掉 :data:`_STOPWORDS` 里的疑问词/虚词（全部被去掉时退回未过滤结果，避免空查询）。
    """
    tokens = [t for t in _cjk_tokens(query).split() if len(t) >= 2 or _is_cjk(t)]
    kept = [t for t in tokens if t not in _STOPWORDS]
    return kept or tokens


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
    """按 H2/H3 切片，保留小节路径（``父 > 子``）；无小节标题的文档退化成整篇。"""
    chunks: list[tuple[str, str]] = []
    trail: dict[int, str] = {}
    current: str | None = None
    buf: list[str] = []
    h1 = ""

    def flush() -> None:
        if not buf or current is None:
            return
        text = "\n".join(buf)
        while len(text) > MAX_CHUNK_CHARS:
            cut = text.rfind("\n\n", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > _MIN_CHUNK_SPLIT else MAX_CHUNK_CHARS
            chunks.append((current, text[:cut]))
            text = text[cut:]
        if text.strip():
            chunks.append((current, text))

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
        chunks = [(h1 or "正文", body)]
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
    """ob_wiki 全文索引：构建 / 检索 / 分节读取。"""

    def __init__(
        self,
        doc_root: Path | str = DEFAULT_DOC_ROOT,
        *,
        wiki_dirname: str = WIKI_DIRNAME,
        index_filename: str = INDEX_FILENAME,
    ) -> None:
        self.doc_root = Path(doc_root)
        self.wiki_dirname = wiki_dirname
        self.wiki_dir = self.doc_root / wiki_dirname
        self.db_path = self.doc_root / index_filename
        self._lock = threading.RLock()
        self._con: sqlite3.Connection | None = None

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

    # ---- 索引生命周期 ----

    def _fingerprint(self) -> tuple[int, int, int]:
        files = size = 0
        newest = 0
        for path in self.wiki_dir.rglob("*.md"):
            try:
                stat = path.stat()
            except OSError:
                continue
            files += 1
            size += stat.st_size
            newest = max(newest, stat.st_mtime_ns)
        return files, size, newest

    def _connect(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(self.db_path, check_same_thread=False)
            self._con.row_factory = sqlite3.Row
        return self._con

    def ensure(self, *, force: bool = False) -> None:
        """确保索引存在且与语料一致；缺失/过期就重建（全库实测约 5 秒）。"""
        with self._lock:
            if not self.wiki_dir.is_dir():
                raise DocIndexMissing(
                    f"文档库不存在：{self.doc_root}/{self.wiki_dirname}（请先解压 backend/doc/ob_wiki.zip）"
                )
            fingerprint = self._fingerprint()
            if not fingerprint[0]:
                raise DocIndexMissing(f"文档库为空：{self.doc_root}/{self.wiki_dirname}")
            if not force and self.db_path.is_file() and self._is_current(fingerprint):
                return
            started = time.monotonic()
            self._build(fingerprint)
            logger.info("文档索引构建完成：%s（%.1fs）", self.db_path, time.monotonic() - started)

    def _is_current(self, fingerprint: tuple[int, int, int]) -> bool:
        try:
            con = self._connect()
            row = con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            if not row or int(row["value"]) != SCHEMA_VERSION:
                return False
            row = con.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
        except (sqlite3.Error, ValueError):
            return False
        if not row:
            return False
        stored = tuple(int(x) for x in str(row["value"]).split(","))
        if stored != fingerprint:
            return False
        # 版本号/指纹都可能被人为改回（或历史索引是旧结构建的）：真查一次列名，
        # 缺列就按过期处理并重建，否则检索会以 "no such column" 报错而不是自愈。
        try:
            con.execute("SELECT path, kind, mode, version, disp_title, section, body FROM chunks LIMIT 0")
        except sqlite3.Error:
            return False
        return True

    def _build(self, fingerprint: tuple[int, int, int]) -> None:
        """先建到临时文件再原子替换，避免并发读到写了一半的索引。"""
        tmp_path = self.db_path.with_name(self.db_path.name + ".tmp")
        if self._con is not None:
            self._con.close()
            self._con = None
        for stale in (tmp_path,):
            if stale.exists():
                stale.unlink()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(tmp_path)
        try:
            con.execute(
                "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)"
            )
            con.execute(
                "CREATE TABLE chunks("
                "id INTEGER PRIMARY KEY, path TEXT, kind TEXT, mode TEXT, version TEXT, "
                "disp_title TEXT, title TEXT, keywords TEXT, section TEXT, body TEXT)"
            )
            con.execute(
                "CREATE VIRTUAL TABLE ft USING fts5("
                "title, keywords, section, body, content='chunks', content_rowid='id', "
                "tokenize='unicode61')"
            )
            rows: list[tuple[Any, ...]] = []
            fts_rows: list[tuple[Any, ...]] = []
            next_id = 1
            for path in sorted(self.wiki_dir.rglob("*.md")):
                rel = path.relative_to(self.wiki_dir).as_posix()
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:  # 单篇读失败不该让整库建不起来
                    logger.warning("跳过无法读取的文档 %s: %s", rel, exc)
                    continue
                meta, body = _parse_frontmatter(text)
                stem = _clean_title(path.stem)
                disp_title = _clean_title(meta.get("title") or stem)
                # 索引用的标题列：标题 + 摘要 + 文件名，三者都是强检索信号
                index_title = " ".join(
                    part for part in (disp_title, stem, meta.get("description", "")) if part
                )
                keywords = _clean_keywords(meta.get("keywords", ""))
                mode = _detect_mode(rel)
                kind = "nav" if path.name.lower() in _NAV_FILENAMES else "doc"
                for section, chunk in _split_chunks(body):
                    rows.append(
                        (next_id, rel, kind, mode, _detect_version(section, rel), disp_title,
                         index_title, keywords, section, chunk)
                    )
                    fts_rows.append(
                        (next_id, _cjk_tokens(index_title), _cjk_tokens(keywords),
                         _cjk_tokens(section), _cjk_tokens(chunk))
                    )
                    next_id += 1
            con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            con.executemany(
                "INSERT INTO ft(rowid, title, keywords, section, body) VALUES (?,?,?,?,?)",
                fts_rows,
            )
            con.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,)
            )
            con.execute(
                "INSERT INTO meta(key, value) VALUES ('fingerprint', ?)",
                (",".join(str(x) for x in fingerprint),),
            )
            con.execute(
                "INSERT INTO meta(key, value) VALUES ('built_at', ?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"),),
            )
            con.commit()
        finally:
            con.close()
        os.replace(tmp_path, self.db_path)

    # ---- 检索 ----

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        mode: str = "",
        version: str = "",
        include_index: bool = False,
    ) -> list[dict[str, Any]]:
        """检索文档小节。

        ``include_index=True`` 让导航页（``index.md`` / ``README.md``）按正常排序参与，适合
        「有哪些 / 包含哪些 / 怎么分类」这类要清单的提问；默认只把它们放在重降权的位置。
        """
        tokens = _query_tokens(query)
        if not tokens:
            raise DocIndexError("query 不能为空")
        limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
        self.ensure()
        # 提问里自带模式/版本时自动消歧（文档库有 MySQL/Oracle 双份同名文档，混着答会答错）。
        # 只当提问**明确写了**才过滤，且过滤后若一条不剩就退回不过滤的检索。
        auto_mode = "" if mode else _detect_mode(query)
        found = None if version else _VERSION.search(query)
        auto_version = found.group(1) if found else ""
        use_mode, use_version = mode or auto_mode, version or auto_version
        with self._lock:
            entries = self._rank(tokens, query, limit, use_mode, use_version, include_index)
            if not entries and (auto_mode or auto_version):
                entries = self._rank(tokens, query, limit, mode, version, include_index)
        return _finalize(entries, limit, prefix=self.wiki_dirname)

    def _rank(
        self,
        tokens: Sequence[str],
        query: str,
        limit: int,
        mode: str,
        version: str,
        include_index: bool,
    ) -> list[dict[str, Any]]:
        """三路召回合并后按统一公式排序，而不是「先命中就早退」的硬分层。

        统一公式（都折算到「-bm25，越大越相关」口径）：
        ``score = -bm25 + 全词命中奖励 - 同义词惩罚 + 版本命中奖励 - 导航小节惩罚``

        为什么要合并而不是分层早退：整句 AND 能命中「相关文档」这类链接清单——它把相邻文档
        的标题整句抄了下来，却没讲答案，一旦早退就会把真正讲该问题的小节挤出结果（实测
        「MySQL 模式的事务隔离级别」丢掉了正文，只剩链接清单）。
        """
        doc_kinds: tuple[str, ...] = ("doc", "nav") if include_index else ("doc",)

        def fetch(groups, n, kinds=doc_kinds):
            return self._match(groups, limit=n, mode=mode, version=version, kinds=kinds)

        rows: list[tuple[sqlite3.Row, float]] = [(r, AND_ALL_TERMS_BONUS) for r in fetch([tokens], limit * 2)]
        rows += [(r, 0.0) for r in fetch([[t] for t in tokens], limit * 3)]
        rows += [(r, -SYNONYM_PENALTY) for r in self._match_expanded(query, limit, mode, version, doc_kinds)]
        if not include_index:
            # 导航页（index.md / README.md）单独一路、重降权召回：既不能硬排除（「OceanBase
            # 数据库包含哪些分类」的最佳答案就是根 README.md），也不能不降权（bm25 会让它挤掉正文）。
            rows += [
                (r, -NAVIGATION_FILE_PENALTY)
                for r in fetch([[t] for t in tokens], limit, ("nav",))
            ]

        best: dict[tuple[str, str], dict[str, Any]] = {}
        for row, bonus in rows:
            key = (row["path"], row["section"])
            entry = _entry(row, bonus, version, tokens)
            if key not in best or entry["score"] > best[key]["score"]:
                best[key] = entry
        return sorted(best.values(), key=lambda e: e["score"], reverse=True)

    def _match(
        self,
        token_groups: Sequence[Sequence[str]],
        *,
        limit: int,
        mode: str = "",
        version: str = "",
        kinds: Sequence[str] = ("doc",),
    ) -> list[sqlite3.Row]:
        """token_groups 之间 OR，组内 AND（传单组即纯 AND 检索）。"""
        clauses: list[str] = []
        for group in token_groups:
            if not group:
                continue
            clauses.append("(" + " AND ".join(f'"{t}"' for t in group) + ")")
        if not clauses:
            return []
        sql = (
            "SELECT c.path, c.kind, c.section, c.disp_title, c.mode, c.version, c.body, "
            "bm25(ft, ?, ?, ?, ?) AS score "
            "FROM ft JOIN chunks c ON c.id = ft.rowid WHERE ft MATCH ?"
        )
        sql_args: list[Any] = [*_BM25_WEIGHTS, " OR ".join(clauses)]
        if kinds:
            # 导航页（index.md / README.md）是纯指路内容、又短又堆关键词，bm25 天然压过正文。
            # 默认只召回正文（kinds=("doc",)），导航页由 _rank 单独降权召回，见那里的说明。
            sql += f" AND c.kind IN ({','.join('?' * len(kinds))})"
            sql_args.extend(kinds)
        if mode:
            sql += " AND (c.mode = ? OR c.mode = '')"
            sql_args.append(_normalize_mode(mode))
        if version:
            sql += " AND (c.version = '' OR c.version LIKE ?)"
            sql_args.append(f"{version.strip().lstrip('Vv')}%")
        # bm25 越小越相关
        sql += " ORDER BY score LIMIT ?"
        sql_args.append(max(limit, 1))
        try:
            return self._connect().execute(sql, sql_args).fetchall()
        except sqlite3.Error as exc:
            raise DocIndexError(f"检索失败：{exc}") from exc

    def _match_expanded(
        self, query: str, limit: int, mode: str, version: str, kinds: Sequence[str] = ("doc",)
    ) -> list[sqlite3.Row]:
        groups: list[list[str]] = []
        for key, aliases in _SYNONYMS.items():
            if key not in query:
                continue
            for alias in aliases:
                alias_tokens = _query_tokens(alias)
                if alias_tokens:
                    groups.append(alias_tokens)
        if not groups:
            return []
        return self._match(
            groups,
            limit=limit,
            mode=mode,
            version=version,
            kinds=kinds,
        )

    # ---- 分节读取 ----

    def read(
        self,
        user_path: str,
        *,
        section: str = "",
        max_chars: int = DEFAULT_READ_CHARS,
    ) -> dict[str, Any]:
        wiki_rel = self.to_wiki_path(user_path)
        max_chars = max(200, min(int(max_chars or DEFAULT_READ_CHARS), MAX_READ_CHARS))
        self.ensure()
        with self._lock:
            rows = self._connect().execute(
                "SELECT section, disp_title, body FROM chunks WHERE path = ? ORDER BY id",
                (wiki_rel,),
            ).fetchall()
        if not rows:
            # 索引里没有（例如新加的文档且索引未刷新）：退回直接读文件
            text = self._resolve(wiki_rel).read_text(encoding="utf-8", errors="replace")
            return {
                "path": self.to_doc_path(wiki_rel),
                "title": _clean_title(Path(wiki_rel).stem),
                "section": "",
                "text": text[:max_chars],
                "truncated": len(text) > max_chars,
                "total_chars": len(text),
            }
        title = rows[0]["disp_title"] or _clean_title(Path(wiki_rel).stem)
        sections = [{"section": r["section"], "chars": len(r["body"])} for r in rows]
        total = sum(len(r["body"]) for r in rows)
        if section:
            picked = [r for r in rows if section.strip().lower() in (r["section"] or "").lower()]
            if not picked:
                raise DocPathError(
                    f"该文档没有匹配的小节：{section}；可用小节见 "
                    f"{self.to_doc_path(wiki_rel)} 的 sections 列表"
                )
        else:
            picked = rows
        text = "\n\n".join(r["body"] for r in picked)
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

    def stats(self) -> dict[str, Any]:
        self.ensure()
        with self._lock:
            con = self._connect()
            chunks = con.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            docs = con.execute("SELECT COUNT(DISTINCT path) AS n FROM chunks").fetchone()["n"]
            built = con.execute("SELECT value FROM meta WHERE key = 'built_at'").fetchone()
        return {
            "db": str(self.db_path),
            "docs": docs,
            "chunks": chunks,
            "schema_version": SCHEMA_VERSION,
            "size_mb": round(self.db_path.stat().st_size / 1e6, 1) if self.db_path.is_file() else 0.0,
            "built_at": built["value"] if built else "",
        }


def _is_navigation_section(section: str) -> bool:
    return bool(_NAV_SECTION.search((section or "").rsplit(" > ", 1)[-1].strip()))


def _excerpt(body: str, tokens: Sequence[str], width: int = 220) -> str:
    """正文里以首个命中词为中心的摘要。

    不用 FTS5 的 ``snippet()``：命中词落在 title/section 列时它对正文取不到片段，
    只会返回一堆省略号；而中文双字 token 还会把词切开（``oceanbase-[database]``）。
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


def _entry(row: sqlite3.Row, bonus: float, version: str, tokens: Sequence[str]) -> dict[str, Any]:
    """把一行召回结果折算成「越大越相关」的条目（bm25 越小越相关，取负）。"""
    score = -float(row["score"]) + bonus
    if version and row["version"]:
        score += VERSION_MATCH_BONUS
    if _is_navigation_section(row["section"]):
        score -= NAVIGATION_SECTION_PENALTY
    return {
        "wiki_path": row["path"],
        "kind": row["kind"],
        "section": row["section"],
        "title": row["disp_title"],
        "mode": row["mode"],
        "version": row["version"],
        "score": round(score, 2),
        "snippet": _excerpt(row["body"], tokens),
    }


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


def configure(
    doc_root: Path | str | None = None,
    *,
    wiki_dirname: str = WIKI_DIRNAME,
    index_filename: str = INDEX_FILENAME,
) -> DocIndex:
    """重置默认索引实例（测试与 CLI 用）。"""
    global _default
    with _default_lock:
        _default = DocIndex(
            doc_root or DEFAULT_DOC_ROOT,
            wiki_dirname=wiki_dirname,
            index_filename=index_filename,
        )
        return _default


def get_index() -> DocIndex:
    global _default
    with _default_lock:
        if _default is None:
            _default = DocIndex()
        return _default


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ob_wiki 离线检索索引")
    parser.add_argument("--doc-root", default=str(DEFAULT_DOC_ROOT))
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    parser.add_argument("--stats", action="store_true", help="打印索引统计")
    parser.add_argument("--query", default="", help="检索并打印前 N 条")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--mode", default="", help="按租户模式过滤：MySQL / Oracle / sys")
    parser.add_argument("--version", default="", help="按版本前缀过滤，如 4.2.5")
    parser.add_argument("--include-index", action="store_true", help="把导航页 index.md 也纳入检索")
    parser.add_argument("--read", default="", help="读取文档（配合 --section）")
    parser.add_argument("--section", default="")
    args = parser.parse_args(argv)

    index = configure(args.doc_root)
    if args.rebuild:
        index.ensure(force=True)
    if args.read:
        print(json.dumps(index.read(args.read, section=args.section), ensure_ascii=False, indent=2))
    if args.query:
        hits = index.search(
            args.query,
            limit=args.limit,
            mode=args.mode,
            version=args.version,
            include_index=args.include_index,
        )
        print(json.dumps(hits, ensure_ascii=False, indent=2))
    if args.stats or not (args.query or args.read):
        print(json.dumps(index.stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())