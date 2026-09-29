# P2 混合检索（Hybrid Retrieval）技术方案（v1，已被取代）

> ## ⛔ 本文已被 [`milvus-only-retrieval-design.md`](milvus-only-retrieval-design.md)（v2）取代
>
> **v1 保留作为"被否决方案"的取舍记录，不再作为实现依据。**
>
> - v1 方案：保留 SQLite FTS5 稀疏一路 + 新增 Milvus Lite 稠密一路（**双引擎**）。
> - v2 决策：**删除 FTS5，稀疏与稠密两路全部由 Milvus Lite 承担**（单引擎）。
> - 本文 §3.5「为什么 BM25 仍用 FTS5」是当时的论证；v2 已按决策**放弃该方案的收益**，代价在 v2 §16 逐条列出。
> - v2 的关键不同：Milvus 相关结论来自**真机实测**（29 项探针），本文不少结论是"文档 + wheel 元数据"推测，且已被实测修正（见 v2 §3.5）。
>
> 仍然有效并可复用的部分：附录 A（RRF 理由）、附录 E（rerank 为何先行的实测依据）、§8.4「导航惩罚只扣一次」的语义、§11.4 的质量判据框架。

> 状态：**已被 v2 取代**（原：待评审）
> 范围：`search_docs` 的第一阶段召回，从「纯 BM25」升级为「BM25 + 稠密向量」混合检索
> 相关代码：[`app/agent/doc_index.py`](../../app/agent/doc_index.py)、[`app/agent/tools.py`](../../app/agent/tools.py)、[`app/config.py`](../../app/config.py)、[`backend/eval/`](../README.md)、[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml)

---

## 0. 决策摘要（TL;DR）

| 项 | 决策 |
| --- | --- |
| 检索形态 | 稀疏 BM25（现有 4 路召回，**公式与常数完全不动**）+ 稠密向量召回，**RRF 排序融合**，再由 **API 交叉编码器重排（P2a）** |
| 融合算法 | Reciprocal Rank Fusion（`k=60`），只用排名，避免 bm25 与 cosine 量纲不可比 |
| 向量存储 | **后端可替换**：默认 **Milvus Lite**（嵌入式、无服务，纯 Python，代码可平移到 Milvus Standalone/Distributed/Zilliz）；回退/测试用 **SQLite 侧车 + numpy 精确余弦**。两者共用同一 `VectorIndex` 协议与「主键 = `content_hash` + upsert」语义。见 §5.2 与附录 F |
| 为什么不用 Milvus Standalone/Qdrant/pgvector | 语料仅 **25077 个块**；需要的是「嵌入式 + 可平移」，而不是分布式。**Milvus Lite 与之不冲突**：它的价值在于同 API 平滑迁到集群（P4）与内建标量过滤/upsert；独立服务级方案仍留到 50 万块量级（见 §14.4） |
| 稀疏一路 | **仍是 SQLite FTS5，不用 Milvus 的稀疏 BM25**：保住零回归兜底、避免 IDF 段内统计与分词分叉（见 §3.5） |
| Embedding 来源 | OpenAI 兼容 `/embeddings` 接口（`openai` 依赖**已在 requirements.txt**）。`embedding.model` 必填，建议 `bge-m3` 或 `text-embedding-3-large` |
| 默认路径 | `retrieval.provider: auto`（默认）→ embedding 配置齐全则 hybrid，否则 bm25；`retrieval.rerank: auto`（默认）→ 配了 rerank 端点则重排，否则等价关闭（＝今天行为）。显式写 `hybrid`/`api` 而对应配置缺失 → **启动即失败**（fail fast，对齐 `auth`/`confirm_timeout` 的风格） |
| 运行期降级 | 查询期 embedding 失败/超时 → **本次检索退回 BM25** 并标记 `retrieval_degraded`；**rerank 超时（2 s）/不可用 → 保留 stage-1 顺序**并标记 `rerank_degraded`。任何情况都**不返回 500、不影响 mock 离线演示** |
| 索引版本 | `SCHEMA_VERSION` 2 → **3**（`chunks` 增 `content_hash` 列），旧索引自动重建 |
| 评测 | `run_eval.py` 增 `--retriever bm25\|vector\|hybrid` 与 `--rerank auto\|off\|api`；CI **三通道**（BM25 零回归恒跑、hybrid 走向量缓存 artifact、rerank 走 nightly/密钥通道）。**上线判据见 §11.4** |
| **Rerank（交叉编码器）** | 原列 P3，**据实测收益上限重排为 P2a（先行于 hybrid）**。BM25 top-30 已含答案 **96%（48/50）**，仅 2 条属真召回失败 → 重排序杠杆 > 扩召回。**形态 = API rerank，默认 `auto`，不做本地 ONNX/GPU 分支**。详见附录 E |
| **候选池稳定性** | 实测同一 query 在 `limit=5` 与 `limit=30` 下排序不同（各路召回预算随 `limit` 缩放）→ **P2a 前置修复**：候选池固定 `K=50` 再截断（见 §16 风险表） |
| 预估工作量 | **≈17.5 人日** = T0 1.0 + 评测扩样 1.5 + P2b（hybrid）12.0 + P2a（rerank，API）3.0；**仅做 hybrid = 14.5 人日**（见 §15） |

---

## 1. 背景与现状

### 1.1 现状

`search_docs`（[tools.py:449](../../app/agent/tools.py#L449)）→ `DocIndex.search`（[doc_index.py:451](../../app/agent/doc_index.py#L451)）→ `_rank`（[doc_index.py:482](../../app/agent/doc_index.py#L482)）当前是**纯稀疏检索**，已经做了不少优化：

- SQLite FTS5 external-content 索引（`content='chunks'`），四列 bm25 权重 `(title, keywords, section, body) = (10, 6, 4, 1)`；
- 中文**单字 + 双字预分词**（`_cjk_tokens`），解决 `trigram` 对 2 字词 0 命中；
- 按 H2/H3 切成**小节块**（`_split_chunks`），块上限 1800 字；
- 查询侧丢噪声 token（单 ASCII 字符、41 个疑问词/虚词）；
- 27 组同义词扩展召回（带惩罚，低优先级）；
- 导航页/导航小节降权（`NAVIGATION_FILE_PENALTY=40`、`NAVIGATION_SECTION_PENALTY=12`）与 `include_index` 逃生开关；
- mode/version 自动消歧过滤（MySQL/Oracle 双份同名文档），过滤后为空则回退不过滤；
- `MAX_CHUNKS_PER_PATH=2`、`_excerpt` 自写摘要。

### 1.2 实测基线（本地真语料，2026-09）

```
语料    5146 篇 → 25077 块（23995 doc + 1082 nav / 682 个 nav 文件），索引 67.6 MB
质量    命中率@1 60.00%   命中率@5 82.00%   命中率@10 94.00%   MRR@10 0.704
延迟    FTS 查询本身 P50 = 0.3 ms；search 端到端 P50 ≈ 98 ms
        其中 _fingerprint()（每次检索全库 stat 5146 个文件）P50 = 51.3 ms
```

> **两种基线口径**：上表是**改造前基线**（现网口径 `limit=5`，用于对比与归因）。P2a 把候选池固定为 `K=50`（§16）后，排序不再随 `limit` 变化，数字会有**一次性小幅漂移**（实测 top-5：40 vs 41 条）。**T1 扩样后重测的「固定池基线」**才是此后 CI 门禁与 §11.4-5 的比对口径，写入 [backend/eval/README.md](../README.md)。

### 1.3 问题（为什么需要 dense）

纯 BM25 的三类结构性弱点，已在评测中暴露：

| 现象 | 用例 | 原因 |
| --- | --- | --- |
| 深度 10 内未命中 3 条 | `lock-wait`、`backup-overview`、`version-425` | 提问用词与文档用词不重叠（「如何查看锁等待的 SQL」→ 答案在《查询行锁》）；BM25 只能靠同义词表兜，表是人工维护的 |
| 命中但排 5 名后 7 条 | 分区表设计、字符集规范、OBProxy 参数、终止租户会话… | 长尾问法下词面重合度低 |
| 清单类只有 50% | `include_index=true` 的 `config-overview`/`sysvar-overview` | 总览页与 FAQ 页竞争 |

> 上表的「未命中」以评测的 `deep=10` 为口径；把候选池放宽到 top-30 后只剩 **2 条**真召回失败（`lock-wait`、`version-425`），`backup-overview` 落在 11–20 名——这正是 P2a 重排先行的依据（见附录 E.2）。

向量召回的价值正是**语义近似**（「锁等待的 SQL」↔「查询行锁」），与 BM25 的**精确字面**（`ORA-00942`、`4.2.5`、配置项名）互补。

### 1.4 已有的基础设施（为什么改动可控）

- **单一接缝**：`DocIndex.search()` 返回 `[{path, section, title, mode, version, score, snippet}]`，`read_doc` 与工具层契约**不需要改**，模型侧无感。工具层 `search_docs` 的返回键名保持 `{query, hits, hit_count}`（`hits` 内既有字段只增不改；本方案新增 `sources`/`stage1_rank`/`rerank_score` 等**诊断字段**，UI 与 `ToolTrace` 不展示）。
- **元数据齐全**：`mode`/`version`/`kind`/`section` 已在索引里，向量侧可直接做预过滤与惩罚。
- **质量门禁齐全**：`backend/eval` 50 条用例 + `hit@k`/`MRR` + CI 阈值 + `--json` 报告 diff，天然用于验证「hybrid 是否真的更好」。
- **离线测试范式**：`tests/helpers/scripted_model.py` 已有「注入 stub」的先例，向量测试同样注入**确定性假 embedder**。

---

## 2. 目标与非目标

### 2.1 目标

1. `search_docs` 在自然语言/口语化提问上的召回质量显著提升：**hit@5 82% → ≥88%**，**MRR@10 0.704 → ≥0.78**（判据见 §11.4）。
2. **不劣化**精确字面查询（错误码、版本号、配置项名）与导航页行为（`test_navigation_pages_never_top_body_answers` 必须继续通过）。
3. 检索质量可回归、可对比：同一个评测集上 `bm25 / vector / hybrid` 三种模式可直接出数字。
4. 索引构建可重复、可增量、可缓存：**同一块内容不重复付 embedding 费用**；语料重新解压不触发重算。
5. 失败可降级、可观测：embedding 不可用时检索仍可用，且原因可见。
6. （**P2a**）重排把 **hit@1 从 60% 提升 ≥5 个百分点**，且不劣化 `literal` 类查询（判据见 §11.4-7）。

### 2.2 非目标（本期不做）

- ~~Reranker / cross-encoder 重排~~ **已从非目标提升为 P2a 目标**（原列 P3，据实测收益上限重排，见 §0 与附录 E）：本方案为其预留固定候选池与 top-N 输出。
- 多向量/ColBERT/late-interaction、HyDE、LLM 查询改写（可作 P3+ 的独立实验）。
- **本地模型推理（ONNX / GPU 部署）**：rerank 与 embedding 均走 API，不引入本地模型与其运行时依赖（决策见 §17.1）。
- 迁到 pgvector / Qdrant / **Milvus Standalone·Distributed**（**P4**，语料上量后再评估）。
- 改动 `read_doc` 行为、改动 agent 事件流与 UI。
- 改 BM25 的排序公式与常数（核心约束：**BM25 侧零回归**）。注意候选池预算的收敛（`limit*2/limit*3` → 固定 `K=50`）不属于改公式，但会带来**一次性基线重测**（§1.2 / §11.4-5）。

---

## 3. 关键决策与选型

### 3.1 存储选型

| 方案 | 新增依赖 | 结论 |
| --- | --- | --- |
| **Milvus Lite（嵌入式、单进程内、可平移 Milvus 集群）** | `faiss-cpu`/`grpcio`/`pyarrow`（实测 ~73 MB wheel）+ `numpy`（已有） | ✅ **采用（默认）**，核实结论见附录 F |
| **SQLite 侧车 + numpy 精确余弦** | 0（`numpy` 已在 requirements） | ✅ **保留为回退/测试后端**（`vector_backend: sqlite_numpy`）：零依赖、可离线做两后端一致性单测 |
| sqlite-vec / LanceDB | 1 个包 | 备选；当前规模不必要 |
| pgvector | PG 扩展 + 与 `memory` 解耦问题 | ❌ 本期不采用（见 §3.4） |
| Milvus Standalone / Qdrant / ES | 独立服务 | ❌ 本期不采用；Milvus Lite 已是同一 API 的嵌入式形态，上量后改 URI 即可（见 §14.4） |

**理由量化**：25077 块在 1024 维下 float32 约 **103 MB**——**这个规模不需要 ANN**（近似索引只会牺牲召回）：Milvus Lite 用 `FLAT` / `BRUTE_FORCE` 即精确检索（**numpy 侧实测 3–8 ms；Milvus `FLAT` 延迟预期同量级，仍待实测，见附录 F.6-3**）。选 Milvus Lite 而非 SQLite 侧车，是为了「**同 API 可平移集群** + 内建标量过滤/`upsert` + 目录级 artifact」（附录 F）；保留 SQLite 侧车是为了**零依赖回退**与**离线单测两后端一致性**（int8 仅该后端可用，25.7 MB）。

### 3.2 融合选型

| 方案 | 结论 |
| --- | --- |
| **RRF（排名融合）** | ✅ 采用。bm25 分（可达 90+）与 cosine（0–1）量纲不可比，归一化脆弱 |
| 分数加权和（min-max / z-score 归一） | ❌ 需在线估计分布，跨 query 不稳定 |
| 分层早退（先向量后 BM25 或反之） | ❌ 现有代码注释已论证过早退会丢正文（"MySQL 模式的事务隔离级别" 只剩链接清单） |

### 3.3 Embedding 选型

| 项 | 决策 |
| --- | --- |
| 接口 | OpenAI 兼容 `POST /embeddings`（`openai` 包已在 requirements，`base_url` 可指向内部部署） |
| 默认模型 | 中文语料建议 `bge-m3`（1024 维）或 `text-embedding-3-large`；**不设内置默认值，必须显式配置**（避免静默用错模型） |
| 维度 | 由模型决定，写入 meta；配置里 `dims` 仅用于校验与容量预估 |
| 归一化 | 入库前 L2 归一化，检索用点积 = 余弦 |
| 量化 | **仅 `sqlite_numpy` 后端**：`int8`（对称量化）或 `float32`；Milvus Lite 只能用 `FLOAT_VECTOR`（103 MB），索引级压缩用 `HNSW_SQ`（附录 F） |
| 离线测试 | `embedding.provider: fake`（哈希派生确定性向量），仅测试/CI 用 |

### 3.4 为什么本期不接 pgvector

`memory.enabled` 是可选能力（PG 不可达时后端照常启动、`/api/threads*` 返回 503、前端隐藏侧栏）。若向量检索复用 memory 连接池，就会出现「关掉 memory → 文档检索降级」的耦合，破坏 README 承诺的降级契约。若为向量单独配 DSN，则等于多一个 PG 实例依赖，反而不如嵌入式方案（Milvus Lite 已覆盖单机场景）。**结论：P4 阶段若已有稳定 PG 且需要多实例共享，再单列 DSN 迁移。**

### 3.5 为什么 BM25 仍用 FTS5，而不是 Milvus 的稀疏检索

Milvus Lite 确实**内建稀疏 BM25 + `SPARSE_INVERTED_INDEX` + 内建融合器**（附录 F.1），所以「两条路都收进 Milvus、只维护一个引擎」在技术上成立。**本期明确不这么做**，理由按权重排序：

1. **会让唯一的保底路径失去「零回归」资格（决定性）。** 现有 BM25 不是教科书 BM25，而是一整套已校准的资产：4 路召回、四列权重 `(10, 6, 4, 1)`、`VERSION_MATCH_BONUS=8` / `AND_ALL_TERMS_BONUS=10` / `SYNONYM_PENALTY=10` / `NAVIGATION_SECTION_PENALTY=12` / `NAVIGATION_FILE_PENALTY=40`、27 组同义词、单双字 CJK 预分词、41 个停用词、**导航页独立召回路径**。Milvus 的稀疏检索只给出「一个 BM25 分数」，以上每一项都得在它之上重建。而 `retriever=bm25` 恰恰是「向量 / rerank / 端点全挂时检索仍可用」的兜底（§10 不变量、§11.4-5 门禁）——把兜底与新引擎（且是 Beta）绑定，等于取消兜底。
2. **中文分词必然分叉。** 自写「单字 + 双字」预分词是为「`trigram` 对 2 字词 0 命中」专门设计的，并与停用词清洗协同；Milvus 的中文分析器走可选 extra `chinese`（jieba，F.1）。分词粒度不同 ⇒ 召回集合不同，`literal` 类（`ORA-00942`、`4.2.5`）行为也随之改变，要保住 `by_tag.literal` 门禁就得再做一轮 tokenizer 对齐。
3. **分数稳定性风险。** Milvus 的 BM25 IDF 为**段内（segment-local）**语义，而 Lite 的存储是 LSM（WAL + 内存表 + 不可变 Parquet 段 + 后台 compaction，F.1）→ 分数可能随段布局/合并漂移；FTS5 是整表统计、可复现。本方案的质量体系是「逐位比对 + CI 门禁」，对「分数随后台任务变化」零容忍。（**本机未实测，已列入 F.6-6**；即便此项不成立，第 1、2、4 条也足以否决。）
4. **会把「检索可用」与「向量基础设施可用」耦合。** BM25 将变成必须依赖 ~73 MB wheel 的 Beta 引擎；`sqlite_numpy` 零依赖回退将**不再能提供 BM25 兜底**；Milvus Lite「每个 data_dir 单进程（文件锁）」的约束会外溢到启动、worker 数与多实例部署（F.2）。
5. **内建融合器用不上。** `RRFRanker`/`WeightedRanker` 只在**两路都在 Milvus 内**时可用，而我们的 RRF 是自研的（附录 A），并且要保留「导航惩罚全局只扣一次」（§8.4）、`include_index` 语义与 `sources`/`stage1_rank` 诊断字段——即便两路都进 Milvus，内建融合器也必须弃用。
6. **当前规模收益为零。** BM25 不是瓶颈：FTS 查询 P50 **0.3 ms**，真正的瓶颈是 `_fingerprint` 的 51.3 ms（由 T0 解决）。统一引擎只省下「少维护一个存储」，代价是上面 1–5。

**何时应重新评估（可证伪判据）**：① 语料上量到几十万块，需要 BM25 也做分布式/多租户；② Milvus 明确提供**全局 IDF**（或 stats 保证）、在 Lite 中提供稳定的中文分析器且稀疏 BM25 转正；③ 需要与服务端检索栈统一（Lite → Standalone 共用同一套 BM25 配置）。届时可把 FTS5 降为「仅兜底」，让 Milvus 稀疏路单独跑 A/B——现有评测器（§11.1）已支持新增一路对比。迁移成本可控：RRF 与惩罚逻辑都是自研的，只需替换 BM25 那一路的取数。

---

## 4. 总体架构

```
                            search_docs(query, limit, mode, version, include_index)
                                              │
                                    DocIndex.search(...)   ← 签名不变（新增 retriever / rerank 形参）
                                              │
                          ┌───────────────────┴───────────────────┐
                          │  候选池构建：固定 K=50（与 limit 解耦） │   ← 修复「排序依赖 limit」
                          └───────────────────┬───────────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
              BM25 路（现有 _rank，公式不变）              Dense 路（新增 _dense_rank）
              4 路召回 + 手调常数                           query embedding（LRU 缓存）
              （导航惩罚开关关闭，见 §8.4）                  向量检索（预过滤 mode/version/kind）
                        │                                           │
                        │                              VectorIndex 协议（§5.2 / 附录 F）
                        │                              ├─ MilvusLiteIndex（默认）
                        │                              └─ SqliteNumpyIndex（回退/测试）
                        └─────────────────┬─────────────────────────┘
                                          ▼
                                 RRF 融合（k=60, w_bm25, w_dense）
                                          ▼
                          ┌───────────────────────────────────────┐
                          │  Rerank 阶段（P2a，可独立开关）        │
                          │  交叉编码器重排 top-30 → 输出 limit     │
                          │  超时 2s / 异常 → 保留 stage-1 顺序     │
                          └───────────────┬───────────────────────┘
                                          ▼
                         后处理：导航惩罚（全局仅一次）→ (path,section) 去重
                                 → MAX_CHUNKS_PER_PATH 上限（移到 rerank 之后）
                                 → limit 截断
                                          ▼
                                     _finalize → 工具层 JSON（契约不变）

  索引侧：ob_wiki.index.db（chunks + FTS5，SCHEMA_VERSION=3；新增 content_hash）
          Milvus Lite data_dir（WAL + Parquet + .idx + manifest）或
          ob_wiki.embeddings.db（SQLite 侧车：content_hash 主键 + 向量）
          —— 两者都是可重建、可缓存的派生 artifact，按 content_hash 复用
```

**两条正交的可开关增强**：`retriever: bm25 | hybrid`（stage-1 召回）与 `rerank: auto | off | api`（stage-2 重排）。组合都合法；`bm25 + rerank off` 即今天的行为。

---

## 5. 数据与存储设计

### 5.1 索引库变更（`ob_wiki.index.db`）

`SCHEMA_VERSION` **2 → 3**，`chunks` 表增加 `content_hash` 列：

```sql
CREATE TABLE chunks(
  id INTEGER PRIMARY KEY,
  path TEXT, kind TEXT, mode TEXT, version TEXT,
  disp_title TEXT, title TEXT, keywords TEXT, section TEXT, body TEXT,
  content_hash TEXT            -- 新增：sha1(归一化后的 body)；向量缓存的键
);
```

- `content_hash` 的输入必须**与嵌入文本一致**（见 §7.2 上下文前缀），否则改前缀会命中旧向量。
- `_is_current`（[doc_index.py:351](../../app/agent/doc_index.py#L351)）已有的**列名自检**会自动让旧索引判定为过期并重建——这也是本方案的安全网。
- 索引库**不存向量本体**（保持 67.6 MB 不变，便于快速重建）；向量在独立存储中（Milvus Lite collection 或 SQLite 侧车，见 §5.2）。

### 5.2 向量存储（后端可替换：Milvus Lite（默认）或 SQLite 缓存 + numpy（回退/测试））

> **Milvus Lite 变体的完整核实结论与映射见 [附录 F](#附录-f-milvus-lite-作为向量后端的核实结论)**。本节描述与后端无关的**语义契约**，两种后端都必须满足。

参考实现（SQLite 侧车）的表结构：

```sql
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);   -- model, dims, quantization, built_at
CREATE TABLE vectors(
  content_hash TEXT PRIMARY KEY,      -- 与 chunks.content_hash 同键
  model TEXT NOT NULL,                -- 模型标识（含版本）
  dims INTEGER NOT NULL,
  vec BLOB NOT NULL                  -- float32；本后端可选 int8（§6.4）
);
```

**设计要点（后端无关）**

1. **跨索引重建存活**：语料重新解压 / `SCHEMA_VERSION` 变化 → 只重建 `index.db`；内容未变的块直接复用向量，**零重复费用**。在 Milvus Lite 上由「主键 = `content_hash` + `upsert`」天然满足。
2. **键是内容哈希**，不是 `chunk_id`（重建后 id 会变）。
3. **模型隔离**：`model` 或 `dims` 变化 → 旧数据视为不匹配，重新计算（不清空，便于回滚对比）。Milvus Lite 用 collection 名或独立字段承载模型标识。
4. 向量存储是**可缓存的 artifact**（§11.3）；Milvus Lite 的 artifact 是**一个目录**（LSM：WAL + Parquet 段 + `.idx` + manifest），不是单文件 —— 原子重建的方式随之变为「临时目录构建 → 重命名/切换指针」。
5. 构建过程可 `--rebuild-vectors` 强制全量；两种后端都要求写入可中断、可续跑。

**磁盘容量**：25077 × 1024 维 × 4 B ≈ **103 MB**（float32，Milvus Lite 的唯一选择）；SQLite 侧车可选 int8（25.7 MB）。

### 5.3 进程内矩阵（仅 `sqlite_numpy` 后端）

Milvus Lite 由引擎自行管理存储与索引（`FLAT` / `BRUTE_FORCE` 即为精确检索），**无需进程内矩阵**；本节只适用于零依赖回退后端。首次需要 dense 检索时惰性构建：

```python
# 伪代码：一次性 join，构造成连续内存块
rows = SELECT c.id, v.vec FROM chunks c JOIN vectors v ON v.content_hash = c.content_hash
       WHERE v.model = ? AND v.dims = ? AND c.kind IN (...)
mat  = np.frombuffer(b"".join(rows.vec), dtype=np.int8).reshape(-1, dims)  # (N, dims)
q    = embed(query)                       # float32, L2 归一化
sims = mat.astype(np.float32) @ q          # int8 反量化可预乘 scale，见 §6.4
```

- 矩阵与 `rowid` 映射缓存在 `DocIndex` 实例上，按 `(index fingerprint, model, dims)` 失效。
- **CI/低内存环境**可配 `retrieval.max_vectors_in_memory`（默认不限）；超限时退化为分片扫描（当前规模不需要）。

### 5.4 缓存键与失效规则（必须与现有指纹机制一致）

> 表头第二列指**向量存储**：`sqlite_numpy` 后端为 `embeddings.db`；`milvus_lite` 后端为 collection（主键同为 `content_hash`），语义相同、操作对应为 `upsert`/按 filter 删除。

| 变化 | index.db（FTS） | 向量存储 |
| --- | --- | --- |
| 语料新增/修改文档 | 重建 | 仅新块计算；旧块复用 |
| `SCHEMA_VERSION` +1 | 重建 | 全量复用（键是内容哈希） |
| 嵌入文本前缀格式变化 | 重建（hash 变） | 全量重算（`content_hash` 变） |
| `embedding.model` / `dims` 变化 | 不重建 | 不删除，按 model 过滤后重算 |
| `embedding.quantization` 变化 | 不重建 | 仅 `sqlite_numpy`：需重编码（本期简单处理：重算） |

---

## 6. 配置设计

### 6.1 新增 `config.yaml` 段

```yaml
retrieval:
  provider: auto          # auto | bm25 | hybrid
  vector_backend: milvus_lite   # milvus_lite（默认）| sqlite_numpy（回退/测试）
  vector_path: ./doc/ob_wiki.milvus  # milvus_lite 时是 data_dir；sqlite_numpy 时是 .db 文件
  rerank: auto            # auto | off | api（**默认 auto**：配了 rerank 端点即重排，否则等价关闭＝今天行为）
  # --- hybrid 生效时的参数 ---
  dense_top_n: 30         # 向量路召回条数（进融合）
  rrf_k: 60
  weight_bm25: 1.0        # RRF 权重
  weight_dense: 1.0
  nav_section_penalty_dense: 12.0   # 仅作用于 dense 条目、且全局只扣一次（见 §8.4）
  query_cache_size: 256   # 查询向量 LRU（条）
  query_timeout_seconds: 5
  pool_k: 50              # 固定候选池，与 limit 解耦（见 §8.2 / §16）
  max_vectors_in_memory: 0          # **仅 sqlite_numpy**：0 = 不限

rerank:                   # 仅当 retrieval.rerank = api 时生效
  base_url: ""            # rerank 端点（Cohere / Jina / 自建 TEI 等）
  api_key: ""
  model: ""               # 如 rerank-multilingual-v3.0 / bge-reranker-v2-m3
  top_n: 30               # 参与重排的候选数
  timeout_seconds: 2      # 超时即保留 stage-1 顺序（§8.6）
  max_passage_chars: 500  # 每条候选送打分的片段长度（§8.6）

embedding:
  base_url: ""            # OpenAI 兼容，如 http://127.0.0.1:9997/v1
  api_key: ""
  model: ""               # 如 bge-m3 / text-embedding-3-large；空 = 未配置
  dims: 0                 # 0 = 由首次响应推断并写入 meta
  batch_size: 64          # 建索引时的批量
  concurrency: 4          # 建索引时的并发请求数
  timeout_seconds: 30
  provider: api           # api | fake（fake 仅测试/CI）
  quantization: float32   # **仅 sqlite_numpy**：float32 | int8（Milvus Lite 无 int8 向量字段，见附录 F）
```

### 6.2 环境变量

`RETRIEVAL_PROVIDER`、`RETRIEVAL_VECTOR_BACKEND`、`RETRIEVAL_VECTOR_PATH`、`RETRIEVAL_RERANK`、`RETRIEVAL_DENSE_TOP_N`、`RETRIEVAL_RRF_K`、`RETRIEVAL_WEIGHT_BM25`、`RETRIEVAL_WEIGHT_DENSE`、`RETRIEVAL_QUERY_CACHE_SIZE`、`RETRIEVAL_QUERY_TIMEOUT_SECONDS`、`RETRIEVAL_POOL_K`、`RETRIEVAL_NAV_SECTION_PENALTY_DENSE`、`RETRIEVAL_MAX_VECTORS_IN_MEMORY`（仅 `sqlite_numpy`）、`RERANK_BASE_URL`、`RERANK_API_KEY`、`RERANK_MODEL`、`RERANK_TOP_N`、`RERANK_TIMEOUT_SECONDS`、`RERANK_MAX_PASSAGE_CHARS`、`EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY`、`EMBEDDING_MODEL`、`EMBEDDING_DIMS`、`EMBEDDING_BATCH_SIZE`、`EMBEDDING_CONCURRENCY`、`EMBEDDING_TIMEOUT_SECONDS`、`EMBEDDING_PROVIDER`、`EMBEDDING_QUANTIZATION`（仅 `sqlite_numpy`）。

沿用 [`_env_nonempty`](../../app/config.py#L161) 语义：**空字符串环境变量不覆盖 YAML 非空值**。

### 6.3 校验规则（`load_settings` 内，风格对齐 `_validate_agent`）

| 规则 | 行为 |
| --- | --- |
| `retrieval.provider: hybrid` 且 `embedding.model` 为空 | **启动失败**（fail fast） |
| `retrieval.provider: auto` 且 embedding 未配置 | 静默 `bm25`，`/api/health` 报 `retrieval: "bm25"` |
| `rrf_k <= 0`、`dense_top_n <= 0`、权重同时为 0 | 启动失败 |
| `retrieval.rerank: api` 且 `rerank.model` 为空 | **启动失败**（fail fast，防「以为在重排其实没有」） |
| `retrieval.rerank: api` 或 `embedding` 已配 model，但对应 `base_url` 为空 | **启动失败**（否则拖到查询期才失败，违背 fail-fast 风格） |
| `retrieval.rerank: auto`（默认）且未配 `rerank.model` | 静默等价 `off`，`/api/health` 报 `rerank: "off"`（**离线/mock 演示不受影响**） |
| `retrieval.rerank` 不在 `{auto, off, api}`、`rerank.top_n <= 0`、`rerank.timeout_seconds <= 0` | 启动失败 |
| `vector_backend: milvus_lite` 但 `pymilvus` 不可导入 | **启动失败**，日志提示 `pip install "pymilvus[milvus-lite]"` 或改 `sqlite_numpy` |
| `vector_backend: milvus_lite` 且显式设置 `embedding.quantization: int8` | 启动失败（Milvus Lite 无 int8 向量字段，见附录 F） |
| `quantization` 不在 `{int8, float32}`（仅 `sqlite_numpy`） | 启动失败 |
| `embedding.provider: fake` 且非测试环境 | 允许但 `WARNING`（防止误用假向量上线） |

### 6.4 int8 量化细节（仅 `sqlite_numpy` 后端）

Milvus Lite 无 int8 向量字段（附录 F），本节不适用；其索引级压缩用 `HNSW_SQ`。对称量化：`scale = max(|v|) / 127`，`q = round(v/scale)`；检索时 `sims = (mat @ q_int8) * scale_row`（`scale_row` 预存于矩阵旁的 float 数组）。归一化向量点积落在 `[-1, 1]`，int8 精度足够（实测 1024 维下与 float32 的 top-30 排名差异 <1%，实现后用评测复核）。

---

## 7. 索引构建流程

### 7.1 流程

```
ensure()
 ├─ 语料指纹校验（现有 _fingerprint，已有 51ms 问题 → 见 §12.3 T0 优化）
 ├─ 建 index.db（现有 _build，新增 content_hash 列）
 └─ rebuild_vectors()            ← 新增，仅 hybrid 模式
      ├─ 读 chunks（id, content_hash, disp_title, section, body, kind）
      ├─ **嵌入集合 = `kind='doc'` 且通过 mode/version 过滤**（导航文件永不嵌入，见 §8.3/§8.4）
      ├─ 与向量存储比对：命中缓存的跳过（**增量的全部意义**）
      │    · sqlite_numpy：按 content_hash 比对行
      │    · milvus_lite：按主键 content_hash 查询/upsert（见附录 F）
      ├─ 未命中的按 batch_size 分组 → concurrent 请求 embed → 写入
      ├─ 断点续传：每批结束提交一次；中断后重跑只补缺口
      └─ 收尾：sqlite_numpy 原子替换 .db 文件；milvus_lite 关闭 client 后切换目录
```

### 7.2 嵌入文本的构造（关键质量项）

**不要只嵌入 `body`。** 采用「上下文前缀 + 正文」：

```
{disp_title} > {section} | {body}
```

理由：同一句「设置方法」在不同文档里语义完全不同；小节块脱离标题后向量会漂移。此改动**与 BM25 侧无关**（索引的 `section`/`title` 列已含这些信息），但要求 `content_hash` 基于**同一前缀后的文本**计算，以保证「改了前缀 ⇒ 向量重算」。

> 该前缀对 BM25 侧同样有利（标题/小节名已在索引列中）。若在 P2b 之前单独落地，此处沿用同一格式即可。

### 7.3 幂等与失败语义

| 情况 | 行为 |
| --- | --- |
| 单批 embedding 失败（429/5xx/超时） | 指数退避重试 3 次；仍失败则**中止向量构建**，保留上次可用的向量存储，`/api/health` 报 `retrieval_degraded`；索引与 BM25 检索照常可用 |
| 缓存缺部分块的向量 | **不报错**：dense 路只覆盖有向量的块，RRF 自然降权缺失项（避免「一个块的失败让整个检索不可用」） |
| `embedding.provider: fake` | 由 `content_hash` 派生确定性向量（如 seeded RNG），仅用于单测/CI 融合逻辑验证，不用于质量评测 |
| 构建期间有查询 | `ensure()` 持 `RLock`；构建在临时文件/临时目录完成再切换（`os.replace` / 目录重命名），并发读者看到旧库或新库，绝无半成品 |

### 7.4 建索引命令行

复用现有 CLI 风格（[doc_index.py:748](../../app/agent/doc_index.py#L748)）：

```bash
python -m app.agent.doc_index --rebuild                 # 重建 FTS 索引
python -m app.agent.doc_index --rebuild-vectors         # 增量补向量（默认）
python -m app.agent.doc_index --rebuild-vectors --force # 全量重算
python -m app.agent.doc_index --stats                   # 打印 docs/chunks/vectors/model/dims/覆盖率
```

`stats()` 增加：`vector_backend`、`vectors`（条数）、`vector_coverage`（有向量的块占比）、`embedding_model`、`dims`、`quantization`（仅 `sqlite_numpy`）、`vector_store_size_mb`。

---

## 8. 检索流程

### 8.1 对外签名

```python
def search(
    self,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    mode: str = "",
    version: str = "",
    include_index: bool = False,
    retriever: str = "",          # "" = 用 self.retriever（来自配置）: "bm25" | "hybrid"
    rerank: str = "",             # "" = 用 self.rerank_mode（auto 已在配置层解析为 "api" | "off"）
) -> list[dict[str, Any]]:
```

- 工具层 `search_docs`（[tools.py:449](../../app/agent/tools.py#L449)）**无需改动**（`retriever`/`rerank` 都走配置默认值）；仅评测与单测会显式传参。
- 返回结构**只增不改** → UI、`ToolTrace`、审计、`read_doc` 全部无感（诊断字段见 §8.5/§8.6）。

### 8.2 伪代码

```python
def search(query, *, limit=5, mode="", version="", include_index=False,
           retriever="", rerank=""):
    tokens = _query_tokens(query)                      # 现有查询侧清洗，dense 路不用 token 但 bm25 用
    retriever = retriever or self.retriever            # "bm25" | "hybrid"
    rerank = rerank or self.rerank_mode                # "api" | "off"（auto 已在配置层解析为二者之一）
    pool_k = self.cfg.pool_k                           # 固定 50，**与 limit 解耦**（修 §16 的 limit 依赖）
    self.ensure()
    auto_mode = "" if mode else _detect_mode(query)
    found = None if version else _VERSION.search(query)
    use_mode, use_version = mode or auto_mode, version or (found.group(1) if found else "")

    with self._lock:
        # ---- stage-1：候选池（固定 K，绝不用 limit 缩放）----
        # 导航惩罚：bm25 路径沿用 _rank 内既有实现（保「逐位一致」）；hybrid 路径关闭内部惩罚，
        # 改由下面的后处理统一扣一次 —— 两者互斥，绝不重复计分（见 §8.4）
        nav_penalty_done = retriever == "bm25"
        bm25 = self._rank(tokens, query, pool_k, use_mode, use_version,
                          include_index, apply_nav_penalty=(retriever == "bm25"))
        if retriever == "bm25":
            fused = bm25
        else:
            try:
                dense = self._dense_rank(query, top_n=self.cfg.dense_top_n, mode=use_mode,
                                         version=use_version, include_index=include_index)
            except VectorUnavailable as e:                  # embedding 超时/限流/存储不可用
                logger.warning("向量召回不可用，本次退回 BM25：%s", e)
                self._mark_degraded(e)
                fused = self._rank(tokens, query, pool_k, use_mode, use_version,
                                   include_index, apply_nav_penalty=True)
                nav_penalty_done = True
            else:
                fused = _rrf([("bm25", bm25), ("dense", dense)],
                             weights={"bm25": w_bm25, "dense": w_dense}, k=rrf_k)

    # ---- 后处理 1：导航惩罚全局只扣一次 + 去重（不截断，先给 rerank 一个完整池）----
    if not nav_penalty_done:
        fused = _apply_nav_penalties(fused, include_index=include_index)
    fused = _dedupe(fused)

    # ---- stage-2：Rerank（可关）----
    if rerank == "api" and fused:
        try:
            fused = self._rerank(query, fused[: self.cfg.rerank_top_n])   # 默认 top-30
        except RerankUnavailable as e:                                     # 2s 超时/模型不可用
            logger.warning("rerank 不可用，保留 stage-1 顺序：%s", e)
            self._mark_rerank_degraded(e)

    # ---- 后处理 2：每篇上限 + limit 截断（顺序很关键，见 §8.6）----
    return _finalize(fused, limit)
```

> 注意 `_finalize` 的 `MAX_CHUNKS_PER_PATH=2` 必须**在 rerank 之后**执行：如果先截断，正确答案所在文档的第二个小节就进不了池，rerank 无从救回（附录 E.3）。

### 8.3 Dense 路细节

**主路径（默认后端 Milvus Lite）**：只调协议，不碰存储实现。

```python
def _dense_rank(self, query, *, top_n, mode, version):
    q = self._embed_query(query)          # LRU 缓存；失败抛 VectorUnavailable
    flt = _scalar_filter(mode=mode, version=version, kind="doc")   # 与 BM25 侧同一套 _mode_ok/_version_ok
    hits = self._vectors.search(q, top_n=top_n, filter=flt)        # Milvus: 标量 filter + FLAT（精确）
    rows = self._fetch_chunks([h.chunk_id for h in hits])          # 一次 IN 查询
    return [_entry_dense(row, h.score, query_tokens) for row, h in zip(rows, hits)]
```

- `VectorIndex.search(vec, *, top_n, filter)` 是**后端无关协议**：`MilvusLiteIndex` 用标量过滤表达式 + `FLAT`/`BRUTE_FORCE`（精确检索，见附录 F.2）；`SqliteNumpyIndex` 在内存矩阵上做布尔掩码 + 点积（回退/测试用，见下）。
- **`kind` 恒定过滤为 `"doc"`**：导航文件**永不嵌入**（嵌入集合见 §7.1 / §12.1），因此 dense 路在任何情况下都不返回 `kind='nav'`；`include_index=true` 的导航页只由 **BM25 路**提供，融合后统一受惩罚（§8.4）。
- **mode/version 过滤必须与 BM25 侧语义一致**（`c.mode = ? OR c.mode = ''`；`version = '' OR version LIKE '4.2.5%'`）。复用同一个 `_mode_ok`/`_version_ok` 辅助函数，避免两条路行为漂移。
- **过滤后为空不做「回退不过滤」**：dense 侧为空就让 RRF 只用 BM25（回退过滤反而会引入错误模式的答案）。
- **导航小节（「相关文档/参见/更多信息/附录」）不排除，但受惩罚**（§8.4）——排除会丢掉「附录」里的有效正文。

<details><summary>回退后端 <code>SqliteNumpyIndex</code> 的实现（仅测试/零依赖场景）</summary>

```python
def _dense_rank_numpy(self, query, *, top_n, mode, version):
    mat, scales, meta_rows = self._vector_matrix()   # 惰性构建 + 缓存；**仅回退后端有**
    q = self._embed_query(query)
    mask = np.array([_mode_ok(r.mode, mode) and _version_ok(r.version, version) and r.kind == "doc"
                     for r in meta_rows])
    idx = np.flatnonzero(mask)
    sims = (mat[idx].astype(np.float32) @ q) * scales[idx]
    order = np.argsort(-sims)[:top_n]
    ...
```
</details>

### 8.4 导航惩罚的「只扣一次」策略（防重复计分）

现状：`_entry`（[doc_index.py:681](../../app/agent/doc_index.py#L681)）内部对导航小节扣 12 分，`_rank` 对导航文件路扣 40 分。

hybrid 下的处理：

| 路 | 导航文件（`kind='nav'`） | 导航小节（`_NAV_SECTION`） |
| --- | --- | --- |
| BM25 | 保持现状（单独一路召回 + 40 惩罚），**但 `_entry` 的 12 分惩罚在 hybrid 模式下关闭**（`apply_nav_penalty=False`） | 同上 |
| Dense | **始终排除**（导航文件不嵌入，§8.3/§12.1）→ dense 永不返回 nav | 导航小节（非 nav 文件）纳入，受 12 惩罚 |
| 融合后 | 对条目按 `kind`/`section` **统一扣一次**（12 / 40） | 同 |

- `retriever=bm25` 时 `apply_nav_penalty=True`，**惩罚语义与今天完全相同**（同一常数、同一条路径）。注意：候选池固定为 `pool_k` 后（§16）排序不再依赖 `limit`，因此评测数字相对**改造前**基线会有一次性重测漂移（已实测 top-5 40 vs 41 条）；此后以 T1 重校的**固定池基线**为准（§11.4-5）。
- `retriever=hybrid` 时惩罚在融合后统一施加，避免「BM25 扣 12 + 融合后再扣 12 = 24」。
- 必然回归测试：`test_navigation_pages_never_top_body_answers` 必须继续通过；`retriever=bm25 --rerank off` 的全量评测数字必须与**固定池基线**逐位一致（§11.4-5）。

### 8.5 RRF 公式

```python
def _rrf(runs, *, weights, k=60):
    """runs: [(name, [entry, ...]), ...]；entry 以 (path, section) 为身份。"""
    scores, first_seen, sources = defaultdict(float), {}, defaultdict(set)
    for name, entries in runs:
        for rank, e in enumerate(entries, 1):
            key = (e["path"], e["section"])
            scores[key] += weights[name] / (k + rank)
            sources[key].add(name)
            first_seen.setdefault(key, e)
    out = []
    for key, s in scores.items():
        e = dict(first_seen[key]); e["score"] = round(s, 6)
        e["sources"] = sorted(sources[key]); e["bm25_score"] = ...; e["dense_score"] = ...
        out.append(e)
    return sorted(out, key=lambda e: e["score"], reverse=True)
```

- **诊断字段**：条目上带 `sources`（`"bm25"` / `"dense"` / 两者）与两路原始分，便于评测报告区分「向量救回来的」与「两路都同意的」。`ToolTrace` 不显示，但 `--json` 报告里有。
- `k=60` 是 RRF 常规取值；`weight_*` 初始 1.0/1.0，按评测微调（建议一次只动一个参数）。

### 8.6 Rerank 阶段（P2a，stage-2）

| 项 | 约定 |
| --- | --- |
| 输入 | 融合后的**完整候选池**（`pool_k=50`），取前 `rerank_top_n`（默认 30） |
| 输出 | 重排后的同一批条目（只改顺序与 `score`），再由 `_finalize` 施加每篇上限与 `limit` 截断 |
| 打分文本 | `文档标题 > 小节路径` + 300–500 字片段（**不是**整块 1800 字，见附录 E.3） |
| 模型 | **API rerank（默认，且不做本地 ONNX / GPU 分支）**：Cohere / Jina rerank 或自建 TEI，由 `rerank.base_url/api_key/model` 配置 |
| 超时 | `rerank.timeout_seconds`（默认 2 s）→ **保留 stage-1 顺序**，不报错、不改返回值 |
| 开关 | `retrieval.rerank: auto \| off \| api`（**默认 `auto`**：配了 `rerank.model` 即启用，否则等价 `off`＝今天行为）；`run_eval.py --rerank auto\|off\|api` 可覆盖 |
| 诊断字段 | 条目带 `stage1_rank`、`rerank_score`，`--json` 报告用于区分「rerank 前移的」与「被压下的」 |
| 零回归要求 | `bm25 + rerank off` 必须与今天逐位一致；`literal` 类查询不得劣化（§11.4） |
| 顺序依赖 | `MAX_CHUNKS_PER_PATH` 与 `limit` 的截断**必须在 rerank 之后**——否则同文档的第二小节进不了池 |

---

## 9. 接口与代码改动清单

| 文件 | 改动 | 备注 |
| --- | --- | --- |
| [`app/config.py`](../../app/config.py) | 新增 `RetrievalConfig`、`RerankConfig`、`EmbeddingConfig` 三个 dataclass；`Settings` 增字段；`load_settings` 解析 + `_validate_retrieval` 校验（含 Milvus Lite 依赖探测） | 沿用 dataclass + `_env_nonempty` 风格 |
| `app/agent/embedding.py`（新） | `EmbeddingClient` 协议 + `ApiEmbeddingClient`（openai/httpx）+ `FakeEmbeddingClient`；批量、并发、重试、维度推断、归一化 | 单文件，便于用 `FakeEmbeddingClient` 单测 |
| `app/agent/vector_store.py`（新） | `VectorIndex` 协议 + `MilvusLiteIndex`（主键 `content_hash` + `upsert`、标量过滤、临时目录构建）+ `SqliteNumpyIndex`（回退/测试：哈希比对、增量补缺、int8 编解码、内存矩阵） | 与 `doc_index.py` 解耦，可独立测试 |
| `app/agent/rerank.py`（新） | `Reranker` 协议 + `ApiReranker`（默认；无本地 ONNX 分支）；批量打分、2 s 超时、`RerankUnavailable` | **P2a**；单测用 `FakeReranker`（无网络） |
| `tests/helpers/fake_reranker.py`（新） | 确定性假重排器（按预设分或关键词分） | 无网络 |
| [`app/agent/doc_index.py`](../../app/agent/doc_index.py) | `chunks` 增 `content_hash`；`SCHEMA_VERSION=3`；`search` 增 `retriever`/`rerank` 形参且**候选池固定 `pool_k=50`**；`_rank`/`_entry` 增 `apply_nav_penalty`；新增 `_dense_rank`（走 `VectorIndex.search` 协议）、`_rrf`、`_apply_nav_penalties`、`_rerank`、`_embed_query`（LRU）、`_vector_matrix`（**仅 `sqlite_numpy` 回退用**）；`_finalize` 的每篇上限**移到 rerank 之后**；`stats()` 扩展；CLI 增 `--rebuild-vectors` | 主战场；BM25 默认行为不变 |
| [`app/agent/tools.py`](../../app/agent/tools.py) | `search_docs` 增加 `VectorUnavailable` / `RerankUnavailable` → `hint` 分支（可选）；**入参 schema 不变** | 契约稳定 |
| [`app/main.py`](../../app/main.py) | 启动时（hybrid 且 model 已配）可选预热：构建/校验向量索引；rerank 端点不可达仅告警 | 参考 memory 的降级处理 |
| [`app/api/chat.py`](../../app/api/chat.py) | `/api/health` 增 `retrieval`、`retrieval_degraded`、`embedding_model`、`vector_coverage`、`rerank`、`rerank_degraded` | 前端 `HealthBadge` 可不改（多余字段被忽略） |
| [`backend/eval/run_eval.py`](../run_eval.py) | `--retriever`、`--rerank auto\|off\|api`、`--min-hit1`（rerank 专用门禁）、`--dense-top-n`、`--rrf-k`、`--weight-*`；报告增 `by_source`、`rerank_moved`（前移/后移的用例） | 评测驱动 |
| [`tests/test_doc_index.py`](../../../tests/test_doc_index.py) | 增融合单测（假 embedder）、导航惩罚只扣一次、增量复用、**候选池与 `limit` 解耦** | 合成语料，快 |
| `tests/test_rerank.py`（新） | 假重排器的顺序变更、2 s 超时降级、每篇上限后移、`literal` 不劣化 | 无网络 |
| `tests/test_vector_store.py`（新） | Milvus Lite / SQLite 两后端一致性、缓存命中/失效/量化往返/损坏库恢复 | 纯单测 |
| `tests/helpers/fake_embeddings.py`（新） | 确定性假 embedder（`hash → 单位向量`） | 无网络 |
| [`README.md`](../../../README.md) / [`README_ZH.md`](../../../README_ZH.md) | 配置表增 `retrieval.*` / `embedding.*`；部署步骤增「构建向量」；工具说明更新 | 双份同步 |
| [`backend/config.example.yaml`](../../config.example.yaml) / [`.env.example`](../../.env.example) | 增示例项 | — |

---

## 10. 失败处理、降级与可观测性

| 场景 | 行为 | 可观测点 |
| --- | --- | --- |
| embedding 未配置 + `provider: auto` | 走 BM25 | `/api/health` → `retrieval: "bm25"` |
| embedding 未配置 + `provider: hybrid` | **启动失败** | 启动日志明确报错 |
| 查询期 embedding 超时/限流/5xx | 本次检索 BM25 结果照常返回 | 日志 `WARNING` + `/api/health` → `retrieval_degraded: true` + `retrieval_error`（首次原因） |
| 向量缓存缺部分块 | dense 路覆盖子集，RRF 自动降权 | `stats().vector_coverage` |
| 向量存储损坏/不可读（`sqlite_numpy` 的 `.db` 或 Milvus Lite 的 data_dir） | 视为「无向量」，可被 `--rebuild-vectors` 重建 | 日志 `ERROR` + health 标记 |
| 向量维度与配置不符 | 视为缓存不匹配，触发重建 | 日志 |
| 进程内存不足（矩阵过大） | 按 `max_vectors_in_memory` 分片扫描（仅 `sqlite_numpy` 后端） | 日志 |
| **rerank 超时（2 s）/端点不可达/限流** | 本次检索**保留 stage-1 顺序**（含融合结果），不报错 | 日志 `WARNING` + `/api/health` → `rerank_degraded: true` |
| Milvus Lite 被多进程打开（文件锁） | 启动期即报错并拒绝启动（与 `--workers 1` 约束一致） | 启动日志明确提示 worker 数 |

**不变量**：任何向量 / rerank 相关故障都**不得**让 `search_docs` 返回错误或 500；`bm25 + rerank off` 是永远的兜底（这也是它必须零回归的原因）。

---

## 11. 评测与验收

### 11.1 评测器扩展

```bash
# stage-1 召回对比
python backend/eval/run_eval.py --retriever bm25    --json /tmp/bm25.json
python backend/eval/run_eval.py --retriever vector  --json /tmp/vector.json
python backend/eval/run_eval.py --retriever hybrid  --json /tmp/hybrid.json
# stage-2 重排对比（可叠加在任一 retriever 上）
python backend/eval/run_eval.py --retriever bm25 --rerank api  --json /tmp/bm25-rr.json
python backend/eval/run_eval.py --retriever hybrid --rerank api --json /tmp/hybrid-rr.json
```

新增输出：
- `by_source`：条目来源分布（`bm25` / `dense` / `both`）；
- `rescued`：BM25 未命中而 hybrid 命中的用例 id 列表（**最有价值的诊断**）；
- `broken`：BM25 命中而 hybrid 未命中的用例 id（**必须逐条解释或修掉**）；
- `by_tag` 增加 `literal`（精确字面：错误码/版本/配置项名）用于「不得劣化」门禁。

### 11.2 评测集扩样（前置任务）

50 条样本上 ±1 条 = 2% 波动，无法支撑「提升」结论（且**目前仅 3 条带 tag**：`list`×2、`hard`×1）。**先扩到 150–200 条**，并补齐 tag：

- `literal`（≥20 条）：错误码、版本号、配置项名、系统变量名、SQL 语法；
- `semantic`（≥60 条）：口语化/不重叠用词（现有弱点类）；
- `list`（≥15 条）：`include_index=true` 清单类（**导航页由 BM25 路提供、dense 不覆盖**，因此这一类的收益不应归因于向量召回，主要靠 rerank 与 BM25 内部排序）；
- `hard`（≥15 条）：长尾、多跳（先查 A 再查 B）。

> 实现注意：现有 `Case.tag` 是**单值**字符串（[run_eval.py:54](../run_eval.py#L54)），上述四个 tag **互斥**，`by_tag` 报告按该单值分组；若将来需要「一例多标签」，需先把字段改为 `tags` 列表（在 T1 内决策）。

### 11.3 CI 三通道

```yaml
# 通道 A（恒跑，快，无密钥）：BM25 零回归，必须显式关掉重排（默认 rerank=auto 会去读端点）
- run: python backend/eval/run_eval.py --strict --retriever bm25 --rerank off --min-recall 0.80 --min-mrr 0.68

# 通道 B（hybrid，无密钥）：优先用缓存好的向量 artifact；miss 时按需降级
- uses: actions/cache@v4
  with:
    path: |
      backend/doc/ob_wiki.embeddings.db
      backend/doc/ob_wiki.milvus
    # 键 = 语料 + 模型名（模型名在此 workflow 里以 vars 形式声明，避免哈希键依赖运行时 env）
    key: vec-${{ hashFiles('backend/doc/ob_wiki.zip') }}-${{ vars.EMBEDDING_MODEL }}
- run: python -m app.agent.doc_index --rebuild-vectors     # 缓存 miss 时用 secret 里的 key 现算
  env: { EMBEDDING_API_KEY: '${{ secrets.EMBEDDING_API_KEY }}' }
- run: python backend/eval/run_eval.py --retriever hybrid --rerank off --min-recall 0.88 --min-mrr 0.78

# 通道 C（rerank，仅有密钥的仓库/定时任务）：逻辑正确性由 FakeReranker 单测覆盖，
# 这里跑真端点的质量门禁；无密钥时该通道跳过，不阻塞 PR
- run: python backend/eval/run_eval.py --retriever hybrid --rerank api --min-recall 0.88 --min-mrr 0.78 --min-hit1 0.65
  env: { RERANK_API_KEY: '${{ secrets.RERANK_API_KEY }}' }
```

配套：新增 `.github/workflows/refresh-vectors.yml`（nightly / 手动）刷新 artifact，避免 PR 上现算 25k 块；本地开发不传密钥时 `provider: auto` + `rerank: auto` 自动落「BM25，不重排」，**离线 mock 演示不受影响**。

### 11.4 上线判据（必须同时满足）

1. 扩样后的评测集上：**hybrid hit@5 ≥ 88%** 且 **MRR@10 ≥ 0.78**；
2. `by_tag.literal` 的 hit@5 **不低于** BM25（相对劣化 ≤1 条）——这一条同时约束 **rerank**；
3. `broken` 列表为空，或每条都在 PR 中给出解释并加评测用例固化；
4. `test_navigation_pages_never_top_body_answers` 通过；
5. `--retriever bm25 --rerank off` 的数字与 **T1 重校后的固定池基线**逐位一致（证明 BM25 打分公式与常数零改动、且排序已与 `limit` 解耦）；相对**改造前** §1.2 基线的差异必须**仅由候选池固定引起**并逐条解释；
6. **延迟门禁按阶段设**（口径见 §12.2；API embedding + API rerank 都是网络调用，原「P50 ≤300 ms」在默认形态下不可达）：
   - `retriever=bm25, rerank=off`（保底路径，含 T0）：P50 ≤ 10 ms；
   - `retriever=hybrid, rerank=off`：P50 ≤ 400 ms、P95 ≤ 900 ms；
   - `retriever=hybrid, rerank=api`（**配了 rerank 端点时的生产默认**）：P50 ≤ 850 ms、P95 ≤ 1.8 s；
7. **rerank 单独判据**（P2a）：`hit@1` 提升 ≥5 个百分点，且 `rerank_moved` 中被后移的正确用例为空（或被解释）。

> 阈值 0.88/0.78 与 `--min-hit1 0.65`（= 改造前 hit@1 60% + 5pp）都是**目标值**，需在 **T1 扩样重校后的固定池基线**上再校准后写进 CI（对齐现有 `--min-recall 0.80` 的「基线往下留一点」做法）；因此**通道 B/C 的阈值依赖 T1 完成**，T1 未落地前只跑通道 A。

---

## 12. 性能预算与成本

### 12.1 建索引（一次性）

| 项 | 估算 |
| --- | --- |
| 块数 | 25077 块中 nav 1082 块**永不嵌入**（§8.3）→ 实际嵌入 **23995** 块 |
| 文本量 | 语料 ~22.3 MB（含 Markdown 标记），中文约 **6–8M token** |
| 请求数 | 23995 / batch 64 ≈ **375 次**（并发 4） |
| 时间 | 视端点：自建/内部部署 bge-m3 约 10–30 分钟；托管 API 约 3–10 分钟 |
| 费用 | `text-embedding-3-small` 量级 ≈ **$0.15**；`-large` ≈ $1.0（一次性，之后仅增量） |
| 磁盘 | Milvus Lite：**103 MB**（float32，无 int8 向量字段）；SQLite 侧车：int8 25.7 MB / float32 103 MB |

### 12.2 查询期（每次 `search_docs`）

| 环节 | 现状 | P2b（hybrid，无 T0） | P2b（含 T0） | + P2a（rerank api） |
| --- | --- | --- | --- | --- |
| `_fingerprint`（全库 stat） | 51.3 ms | 51.3 ms | **≤1 ms**（缓存/TTL） | 同左 |
| FTS 查询 | 0.3 ms | 0.3 ms | 0.3 ms | 0.3 ms |
| 查询 embedding（网络） | — | **50–300 ms**（端点相关，**新的主导项**） | 同左 | 同左 |
| 向量检索 | — | 3–8 ms（Milvus `FLAT`，回退 numpy 点积） | 3–8 ms | 3–8 ms |
| RRF + 后处理 | ~1 ms（现有打分） | ~1 ms | ~1 ms | ~1 ms |
| **Rerank（API，top-30）** | — | — | — | **100–500 ms** |
| **合计 P50** | **≈98 ms** | **≈110–360 ms** | **≈60–310 ms** | **≈160–810 ms** |

要点：
- **两个网络调用是主要代价**：hybrid 加一次 embedding（50–300 ms），默认形态再加一次 API rerank（100–500 ms）。因此 `query_cache_size` LRU、`query_timeout_seconds=5`、`rerank.timeout_seconds=2` 都是硬约束。
- **延迟门禁必须按阶段设（§11.4-6）**：原方案的单一「P50 ≤300 ms」已按阶段拆分——`hybrid + rerank off` 为 ≤400 ms，**配了 rerank 端点时（生产默认）为 P50 ≤850 ms / P95 ≤1.8 s**（上表合计上限 810 ms 落在门禁内）。
- 不需要为「FTS 与 embedding 并行」做优化（FTS 仅 0.3 ms）；rerank 期间 `RLock` 只覆盖索引/向量读取，不阻塞其他请求。
- 若默认形态延迟仍不可接受：调小 `rerank.top_n` / `dense_top_n` / `max_passage_chars`，或把 `retriever` 落回 `bm25`（一条配置，见 §14.2）。
- **T0（修 fingerprint + SQLite pragma）建议在 P2b 之前单独落地**：它是零风险收益，且能让 hybrid 的延迟对照更干净。

### 12.3 与 T0 的关系（强烈建议前置）

`journal_mode=delete`/`mmap_size=0`/`cache_size=2000`（实测值）对只读索引是浪费；`_fingerprint` 每次检索全库 stat 是当前最大单点开销。这两项与检索质量无关、可与 P2b 解耦、且评测数字应**完全不变**，是理想的先行项。

### 12.4 成本小结

- 一次性：**$0.15–1.0**（托管端点）或 10–30 分钟（自建嵌入端点）。
- 增量：仅新/改文档计费（内容哈希缓存）。
- 运行期 embedding：每次检索 1 次调用（未命中 LRU 时）；无向量数据库服务器成本（Milvus Lite 嵌入式）。
- 运行期 rerank：按调用计费，与 `rerank.top_n` × 片段长度相关，量级 **$1–3 / 1000 次检索**；需按所选端点核实。

---

## 13. 安全与合规

- 语料是**公开官方文档**，送往用户配置的 embedding 端点；部署方需知晓该端点（内部部署可完全内网）。这一点与「对话内容会发送到 LLM API」的既有事实一致，[README](../../../README.md) 已有类似披露，建议在文档配置段补一句。
- embedding 端点凭据走 `EMBEDDING_API_KEY` / `.env`（已 gitignore），**不得**写入 `config.example.yaml`。
- 向量存储内含文档向量（可由原文近似重建），视同语料本身密级；`ob_wiki.embeddings.db` 与 Milvus Lite 的 `ob_wiki.milvus/` **目录**都在 `backend/doc/` 下，**已被现有 `.gitignore` 规则 `/backend/doc/*`（`ob_wiki.zip` 除外）覆盖**，无需新增规则——但需在 T8 用 `git check-ignore` 实测确认一次（文件与目录都要测）。
- 向量存储不存原文，只存哈希与向量，泄露面小于索引库本身。
- **rerank 会把候选文档片段（公开语料，默认 ≤`max_passage_chars`=500 字/条）送往 rerank 端点**，与 embedding 同等披露；`RERANK_API_KEY` 同样只走 env，**不得**写入 `config.example.yaml`。

---

## 14. 迁移、灰度与回滚

### 14.1 上线步骤

1. 合并 T0（性能），确认评测数字不变。
2. 扩评测集到 150–200 条，重校 BM25 基线（记录到 [backend/eval/README.md](../README.md)）。
3. **先落 P2a（rerank）**：`RETRIEVAL_RERANK=api`，只对 BM25 候选池重排 → BM25 召回完全不变、收益可独立归因（§11.4-7）。
4. 实现 P2b（默认 `provider: auto` → 现有部署行为不变；显式 `hybrid` 才启用）。
5. **影子模式**：`retriever=hybrid` 但只记录两路排名与 `rescued`/`broken`，不改变返回（用一个临时观测开关或只在评测中跑），先积累真实分布。
6. 灰度：先在单实例把 `RETRIEVAL_PROVIDER=hybrid` 打开，观察延迟与 `/api/health` 的降级标记。
7. 默认化：`provider: auto` + `rerank: auto` 即等价于「配齐端点就 hybrid + 重排」，因此**只需保证 embedding/rerank 配置存在**；随后在 README/config 示例中把 `retrieval`/`embedding`/`rerank` 标为推荐配置。

### 14.2 回滚

- **一行回滚**：`RETRIEVAL_PROVIDER=bm25`（或删掉 `embedding.model`），无需数据迁移、无需重建索引。
- **只关重排**：`RETRIEVAL_RERANK=off`（或删掉 `rerank.model`），保留 hybrid 召回——用于「rerank 端点抖动 / 延迟超标 / 字面查询变差」的单点回退，比整体回滚 BM25 更精准。
- 向量存储可保留（不参与 BM25 路径），也可删除（下次自动重建）。

### 14.3 兼容性

- 旧客户端/前端：`/api/health` 只增字段，`HealthBadge` 忽略未知字段；SSE 事件不变。
- 旧索引文件：`SCHEMA_VERSION=3` 触发自动重建（一次 ~5–8 s），无手工步骤。
- 旧配置：不写 `retrieval`/`embedding`/`rerank` 即 `auto` + 未配置 → 行为与今天完全一致（BM25、不重排、无端点调用）。

### 14.4 P4 迁移预留

`VectorIndex` 与 `EmbeddingClient` 均以协议定义，未来替换为 Milvus Standalone/pgvector/Qdrant 只影响 `MilvusLiteIndex`（改 URI）或 `_dense_rank` 的取数方式；`_rrf`、`_rerank` 与后处理无需改动。

---

## 15. 里程碑与任务拆解

| # | 任务 | 产出 | 估时 |
| --- | --- | --- | --- |
| T0 | **性能前置**：fingerprint 缓存/TTL + SQLite pragma | 检索 P50 降到个位数 ms，评测数字不变 | 1.0 d |
| T1 | 评测集扩样到 150–200 条 + 补 tag（含 `literal`）+ 重校基线 | `retrieval_cases.jsonl`、README 基线 | 1.5 d |
| — | **以下为 P2b（hybrid 召回）** | | |
| T2 | `RetrievalConfig`/`RerankConfig`/`EmbeddingConfig`（含 `vector_backend`、`pool_k`、`rerank: auto`）+ 校验（含 Milvus 依赖探测）+ 示例配置 + README | 配置可读可用，fail-fast 生效 | 1.0 d |
| T3 | `embedding.py`：API/Fake 客户端（批量、并发、重试、归一化） | 单测（假端点） | 1.5 d |
| T4a | `MilvusLiteIndex`：schema（`content_hash` 主键 + `path/section/mode/version/kind` 标量字段）、`upsert` 增量、标量过滤、临时目录构建 | 真 Milvus Lite 上可建可查 | 2.0 d |
| T4b | `SqliteNumpyIndex`：哈希比对、增量补缺、int8、内存矩阵 | 零依赖回退后端 | 1.0 d |
| T4c | 两后端一致性测试（同批块 → 同 top-N；模式/版本过滤等价） | `tests/test_vector_store.py` | 0.5 d |
| T5 | `doc_index.py`：`content_hash` + `SCHEMA_VERSION=3` + 建向量流程 + CLI/`stats` | `--rebuild-vectors` 可用 | 1.5 d |
| T6 | `doc_index.py`：**复用 R1 的固定候选池** + `_dense_rank` + `_rrf` + 导航惩罚只扣一次 + `retriever` 形参 | BM25 公式/常数零改动；融合零回归 | 1.5 d |
| T7 | `run_eval.py --retriever/--rerank` + `rescued`/`broken`/`by_source`/`by_tag.literal` | 模式对比报告 | 1.0 d |
| T8 | CI 三通道 + 向量 artifact 缓存 + nightly 刷新 + `.gitignore` 实测 | CI 绿且 hybrid/rerank 有门禁 | 1.0 d |
| T9 | 影子观测 → 灰度 → 默认化 + 文档同步（中英） | hybrid 上线 | 1.0 d |
| — | **以下为 P2a（rerank）——执行顺序上先行于 P2b**（见 §14.1 / §17.1；表内按阶段分组，非时间顺序） | | |
| R1 | **固定候选池 `pool_k=50` 落地（P2a 前置，归此任务）** + `rerank.py`（`Reranker` 协议 + `ApiReranker`、批量打分、2 s 超时、`RerankUnavailable`） | `bm25 + rerank api` 可跑；**重测并固化固定池基线**（§11.4-5） | 1.0 d |
| R2 | `_rerank` 阶段 + **每篇上限/limit 截断后移** + `/api/health` 标记 | stage-2 可开关，降级正确 | 1.0 d |
| R3 | 评测调参与门禁：`hit@1`、`literal` 不劣化、`rerank_moved` | 满足 §11.4-7 | 1.0 d |
| **合计** | | | **≈17.5 d**（T0 1.0 + T1 1.5 + P2b 12.0 + P2a 3.0）；**仅做 hybrid = 14.5 d**。P2a 比初版少 0.5 d：不做本地 ONNX 分支（无模型打包与 CI 模型缓存） |

### 15.1 测试计划（离线、确定性）

- **融合单测**：`FakeEmbeddingClient`（`content_hash → 单位向量`）构造「BM25 弱、dense 强」的合成语料，断言 hybrid 把正确答案提到前面；断言 `retriever=bm25` 结果与改造前逐位一致。
- **惩罚不重复**：断言 hybrid 下导航小节只被扣一次（构造一个两路都会命中的导航小节）。
- **缓存增量**：改一篇文档后重跑建向量，断言只有该篇的块触发 embed 调用（用计数型 fake 客户端）。
- **量化往返**（仅 `sqlite_numpy` 后端）：int8 编码/解码后 top-30 排名与 float32 一致（合成数据）。
- **候选池与 `limit` 解耦**：断言同一 query 在 `limit=5` 与 `limit=30` 下的**排名前缀一致**（修 §16 那条高风险）。
- **rerank 顺序与降级**：假重排器注入超时/异常 → 断言保留 stage-1 顺序且不报错；断言**每篇上限与 limit 截断在 rerank 之后**生效（同文档第二小节可被救回）。
- **两后端一致性**：同一批块在 `MilvusLiteIndex` 与 `SqliteNumpyIndex` 上 top-N 相同、模式/版本过滤等价。
- **降级**：注入必然超时的客户端，断言返回 BM25 结果、无异常、health 标记被设置。
- **回归**：`tests/test_doc_index.py`、`tests/test_retrieval_eval.py`（真语料）全绿；`test_navigation_pages_never_top_body_answers` 必须通过。

---

## 16. 风险登记

| 风险 | 影响 | 概率 | 缓解 |
| --- | --- | --- | --- |
| 查询期 embedding 延迟/不稳定 | 检索变慢、体验下降 | 中 | 硬超时 5 s + LRU + 失败落 BM25 + 延迟门禁（§11.4-6） |
| 评测集太小 → 误判提升 | 错误上线，长期质量下降 | **高** | T1 前置扩样；`broken` 逐条解释；`literal` 单独门禁 |
| 导航页/导航小节被向量重新顶到第一 | 破坏既有保证 | 中 | dense 默认排除 nav 文件；导航小节惩罚统一扣一次；专项回归断言 |
| 内容哈希与嵌入文本不一致 → 改了前缀却复用旧向量 | 静默质量退化 | 中 | 哈希基于**最终嵌入文本**计算；改格式必须 bump `SCHEMA_VERSION`；单测断言 |
| 模型/端点更换后参数未同步 | 检索质量异常 | 中 | 向量存储的 meta 记录 model/dims（`sqlite_numpy` 为 `embeddings.db` 的 `meta` 表；Milvus Lite 为 collection 名/独立字段），不匹配即重算；`stats()` 暴露 |
| 精确字面查询被稀释 | `ORA-00942` 类问题变差 | 中 | `by_tag.literal` 门禁；RRF 权重可偏向 BM25；必要时对「纯字面 query」直接走 BM25 |
| 依赖外部 embedding 服务成为新单点 | 部署复杂化 | 中 | `auto` 自动降级；`RETRIEVAL_PROVIDER=bm25` 一键回滚；`vector_backend` 可切 `sqlite_numpy`（无本地模型依赖） |
| **排序依赖 `limit`（各路召回预算是 `limit*2`/`limit*3`）** | 候选池不稳定 → 重排序的「池」与上线口径不一致；实测同一 query：`limit=5` 时 hit@5 82%，从 `limit=30` 的排名取 top-5 只有 80% | **高** | P2a 前置：候选池固定 `K=50`（与 `limit` 解耦）后再截断；`retriever` 各模式共用同一池 |
| 交叉编码器压低精确字面结果 | 错误码/版本号类查询变差 | 中 | 固定池 + `by_tag.literal` 门禁；必要时对纯字面 query 跳过 rerank |
| Rerank 延迟高于 hybrid 的 embedding 调用 | agent 多轮检索累计变慢 | 中 | 只重排 top-30、片段截 500 字；2 s 超时即保留 stage-1 顺序；门禁按阶段设（§11.4-6） |
| **两个 API 依赖串联（embedding + rerank），任一抖动都拖慢检索** | 尾部延迟与可用性 | 中 | 各自独立超时与降级（embedding 失败落 BM25；rerank 失败保留 stage-1），**不共用超时预算**；`/api/health` 分别标记 |
| **Milvus Lite 处于 Beta（Development Status 4）**，且 3.x 与旧版 v1 `.db` 格式不兼容 | 升级/回退可能需重建；线上稳定性风险 | 中 | 精确 pin 版本；`vector_backend: sqlite_numpy` 作为等价回退；向量是**可重建的派生数据**，最坏情况重算 |
| Milvus Lite 限制「每个 data_dir 单进程」（文件锁）且写入需串行 | 多 worker 部署下打开失败/写入冲突 | 中 | 与项目既有 `--workers 1` 约束一致；启动显式校验并给出明确报错；文档标注 |
| Milvus Lite 依赖体积（faiss-cpu / pyarrow / grpcio） | 镜像与 CI 体积增加 | 中 | 实测 Linux wheel 合计 ~73 MB；`sqlite_numpy` 后端作为零依赖回退；CI 缓存 wheel |
| 建索引费用/时长失控 | 一次性成本与 CI 时间 | 低 | 内容哈希增量 + artifact 缓存 + nightly 刷新 |
| 内存占用（矩阵随语料增长） | OOM | 低 | **仅 `sqlite_numpy`**：int8 + `max_vectors_in_memory` 分片；Milvus Lite 向量在磁盘、内存由引擎管理 |

---

## 17. 待决问题（Open Questions）

### 17.1 已定（本方案据此定稿，不再讨论）

| 事项 | 结论 |
| --- | --- |
| rerank 形态 | **API rerank，不做本地 ONNX / GPU 分支**（§8.6）；`retrieval.rerank` 默认 `auto` |
| 向量后端 | **Milvus Lite 为默认**（`vector_backend: milvus_lite`），`sqlite_numpy` 仅回退/测试（附录 F） |
| 候选池 | 固定 `pool_k = 50`，与 `limit` 解耦（§8.2 / §16） |
| 延迟门禁 | 按阶段设；生产默认形态 P50 ≤850 ms / P95 ≤1.8 s（§11.4-6） |
| 本地模型 | 不引入（embedding 与 rerank 均走 API） |
| 阶段顺序 | T0 → T1 → **P2a（rerank 先行）** → P2b（§15） |

### 17.2 仍需确认

1. **Embedding 端点与模型**：内部部署 `bge-m3` 还是云端 `text-embedding-3-large`？谁提供端点与配额？（决定延迟与费用）
2. **rerank 端点与模型**（阻塞 R1 的默认值）：托管（Cohere / Jina）还是自建 TEI？`rerank.top_n` 取 30 还是 20（延迟换收益）？是否与 embedding 共用同一端点与密钥？
3. **dims 是否固定** 1024 维，以便缓存跨环境复用？
4. **评测集扩样**由谁负责、「正确答案」的标注依据是什么（当前是路径子串，需人工确认）？
5. **CI 密钥策略**：允许在 CI 注入 embedding / rerank key（有费用与泄露面），还是只依赖缓存 artifact + nightly 刷新？（推荐后者；rerank 的逻辑正确性由 `FakeReranker` 单测覆盖，质量门禁走 nightly 通道 C）
6. **`include_index` 的 list 类**是否单独设阈值与策略（当前 50%，向量未必能解决，可能是切片粒度问题）？
7. **是否精确 pin `milvus-lite` 版本**（3.x 为 Beta——F.1；与旧 v1 数据格式不兼容——F.5）？

---

## 附录 A：为什么 RRF 而不是分数相加（展开）

- bm25 是无界负分（FTS5 返回越小越相关，代码取负后可达 90+，见 `NAVIGATION_FILE_PENALTY` 的注释实测值 91.18）；cosine 归一化后在 `[-1, 1]`。二者不可直接相加。
- 归一化（min-max/z-score）依赖每次查询的候选分布，长尾查询上不稳定，且会让「BM25 的 27 组同义词与导航惩罚」这些精心校准的相对次序失效。
- RRF 只用**排名**，因此：
  - BM25 侧所有既有手调常数**自动保留**（它们的价值体现在 BM25 路内部的排序上）；
  - 向量侧只需「相似度排序正确」，不需要与 BM25 对齐量纲；
  - 调参面收敛为 `rrf_k` + 两个权重，可评测驱动。

## 附录 B：配置样例（完整）

```yaml
llm:
  base_url: https://api.example.com/v1
  api_key: sk-...
  model: gpt-4o-mini

embedding:
  base_url: https://api.example.com/v1
  api_key: sk-...
  model: bge-m3
  dims: 1024
  batch_size: 64
  concurrency: 4
  timeout_seconds: 30
  provider: api
  quantization: float32   # 仅 sqlite_numpy 生效

rerank:
  base_url: https://api.example.com/v1
  api_key: sk-...
  model: rerank-multilingual-v3.0
  top_n: 30
  timeout_seconds: 2
  max_passage_chars: 500

retrieval:
  provider: auto
  vector_backend: milvus_lite
  vector_path: ./doc/ob_wiki.milvus
  rerank: api
  dense_top_n: 30
  rrf_k: 60
  weight_bm25: 1.0
  weight_dense: 1.0
  nav_section_penalty_dense: 12.0
  query_cache_size: 256
  query_timeout_seconds: 5
  pool_k: 50
```

## 附录 C：`/api/health` 变更示意

```json
{
  "status": "ok",
  "ocp_provider": "mock",
  "sql_provider": "mock",
  "llm_configured": true,
  "memory_enabled": false,
  "auth_enabled": false,
  "retrieval": "hybrid",
  "vector_backend": "milvus_lite",
  "embedding_model": "bge-m3",
  "vector_coverage": 0.9987,
  "rerank": "api",
  "rerank_degraded": false,
  "retrieval_degraded": false
}
```

## 附录 D：术语

| 术语 | 含义 |
| --- | --- |
| 稀疏检索 / BM25 | 基于词频与逆文档频率的字面匹配（当前实现） |
| 稠密检索 / dense | 用 embedding 向量做语义近邻 |
| RRF | Reciprocal Rank Fusion，排名倒数加权融合 |
| reranker | 交叉编码器对候选重排（**P2a**，见附录 E） |
| agentic RAG | 由 agent 决定检索时机与多轮追读的 RAG（当前形态） |

---

## 附录 E：Rerank 为何从 P3 提前到 P2a（实测依据）

### E.1 实测：重排序的收益上限远大于扩大召回

在真语料上把检索深度放到 30 后统计**累积命中率**（若重排序完美，`hit@k` 的上限即「答案落在前 k」的比例）：

| 指标 | 实测值 |
| --- | --- |
| 答案已在 top-30 内（**重排序可达上限**） | **96%（48/50）** |
| 答案已在 top-10 内 | 94%（47/50） |
| 答案已在 top-5 内 | **80%（40/50）** |
| 排名分布 | top-5 达标 **40 条**；6–10 名 **7 条**；11–20 名 **1 条**；top-30 之外 **2 条** |

> 口径提示：上表是 **depth=30** 那一次 run 的累积分布（top-5 = 40）。而**生产口径**（现有默认 `limit=5`）测得 `hit@5 = 82%（41/50）`——**两次 run 的 top-5 差 1 条**，正是「各路召回预算随 `limit` 缩放」造成的排序不稳定（§16 高风险行）。这也是 P2a 必须先把候选池固定为 `K=50` 的原因：否则重排用的池与上线口径不是同一个池。

结论：本题库的缺口比例是「**排序问题 8 条 : 召回问题 2 条**」。跨编码器是 stage-2，其收益上限（`hit@5 → 可达 96%`）高于 hybrid 的**独有**收益（只能救 top-30 外那 2 条）。因此**先做 hybrid 而不做 rerank，会把更大的一块收益留在桌上。**

另有 **11 条**用例的正确答案落在 2–5 名（生产口径 `limit=5` 下 `hit@1=60%`、`hit@5=82%`）——这些直接决定 `hit@1`，正是重排序最擅长的一类，无需任何召回改动。

### E.2 原 P3 排序的理由（仍成立，需在 P2a 中处理）

1. **逐对前向成本高**：20–50 对逐对打分；本方案取 **API 形态 ≈100–500 ms**（本地 CPU 方案约 0.5–2 s，已排除）；agent 一轮可能多次 `search_docs`。
2. **块长与模型窗口不匹配**（真正的工程障碍）：现有块上限 1800 字 ≈ 900–1200 中文 token，而 `bge-reranker-base` 窗口仅 512 token。
3. **可能压低精确字面查询**（错误码/版本号/配置项名）。
4. **CI 需要端点密钥**（本地模型方案才需要模型缓存，已排除）；一次改两个机制会让评测数字无法归因（可用 `--rerank off` 单独评，故非阻塞理由）。

### E.3 若采纳 P2a，设计要点

| 项 | 方案 |
| --- | --- |
| 候选池 | **固定 `K=50`**（先修「排序依赖 `limit`」，见 §16），与 `limit` 解耦；各 `retriever` 模式共用 |
| 重排范围 | 池内 top-30 → 输出 `limit` |
| 每篇上限 | `MAX_CHUNKS_PER_PATH=2` **移到 rerank 之后**（否则正确答案的第二个小节进不了池、无法参与竞争） |
| 打分文本 | `文档标题 > 小节路径` + **300–500 字摘要片段**，而非整块 1800 字；若用长上下文模型（`bge-reranker-v2-m3`，8k）可放宽 |
| 模型 | **API rerank（默认，不做本地 ONNX）**：中文语料优先 bge-reranker 系列的托管端点（如 `bge-reranker-v2-m3`）或 Cohere/Jina rerank |
| 降级 | `rerank.timeout_seconds`（默认 2 s）超时/异常 → **保留 stage-1 顺序**（与 hybrid 相同的降级哲学），`/api/health` 标记 |
| 评测 | `--rerank api/off`；`by_tag.literal` 不得劣化；`broken` 逐条解释；预期 `hit@1 60% → 70–85%`、`hit@5 82% → ≤96%`（**待实测**） |

### E.4 修正后的顺序与判据

| 阶段 | 内容 | 位置理由 |
| --- | --- | --- |
| **P2a（先行）** | Rerank（固定池 + 交叉编码器） | 不需要 embedding 端点、不需要向量存储，收益/成本比最高，可最快用评测验证 |
| **P2b（随后）** | Hybrid 召回（本方案主体） | 唯一能解决 top-30 外的召回失败；同时扩大池子，给 rerank 更多可排 |

判据：**池内/池外缺口比例**。本语料为 8:2 → 先 rerank；扩样到 150–200 条后若池外缺口上升（长尾问法变多），P2a 与 P2b 并行。

---

## 附录 F：Milvus Lite 作为向量后端的核实结论

> 结论：**可行，且比最初假设更合适。** 新版 `milvus-lite` 3.x 是**纯 Python 重写**的嵌入式引擎（不再是旧的 C++ 大包），无服务、单进程内使用，且同一套 `pymilvus` 代码可平移到 Milvus Standalone / Distributed / Zilliz Cloud。
> 来源：PyPI 项目页与 wheel 元数据（外部资料，2026-08 发布的 3.2.1）：[milvus-lite · PyPI](https://pypi.org/project/milvus-lite/)、[wheel 文件](https://files.pythonhosted.org/packages/f8/ac/7534ce7526191f776e0d8f28c32ea69f0ae9516b1510a5653eeb5038e1b9/milvus_lite-3.2.1-py3-none-any.whl)

### F.1 核实到的事实

| 项 | 核实结果 |
| --- | --- |
| 形态 | 嵌入式：`MilvusClient("./demo.db")` 即本地启动；也可跑本地 gRPC server（多客户端开发用） |
| 包体 | `milvus_lite-3.2.1-py3-none-any.whl` = **269.6 KB**，纯 Python（126 个文件）；**Requires-Python ≥3.10** |
| 硬依赖 | `faiss-cpu`、`grpcio`、`pyarrow`、`numpy`（`numpy` 项目已有）；可选 extra `chinese`（jieba 分词器） |
| 依赖体积（实测） | Linux x86_64 wheel：`faiss-cpu` 26 MB + `pyarrow` 40 MB + `grpcio` 6.8 MB ≈ **73 MB**（安装后更大） |
| 能力 | 稠密 + 稀疏 BM25 + 混合检索（内建 `WeightedRanker`/`RRFRanker`）；标量/JSON/数组过滤；`upsert`、按 ID/过滤删除；分区；快照 |
| 索引 | `HNSW`、`HNSW_SQ`、`IVF_FLAT`、`IVF_SQ8`、`FLAT`、`BRUTE_FORCE`、`AUTOINDEX`、稀疏 `SPARSE_INVERTED_INDEX`、标量 `INVERTED` |
| 存储布局 | LSM：WAL + 内存表 + 不可变 Parquet 段 + `.idx`，manifest 原子更新 → **是一个目录，不是单文件** |
| 平台 | macOS / Linux / Windows（取决于 `faiss-cpu`、`pyarrow` wheel）；CI 覆盖 Linux 3.10–3.13、macOS 3.12、Windows 3.10 |
| 状态 | Development Status **4 - Beta**；定位「本地开发与小规模负载」，非分布式生产服务 |

### F.2 对本次设计的映射（必须调整的点）

| 设计点 | 原方案（SQLite 侧车） | Milvus Lite 下 |
| --- | --- | --- |
| 量化 | `int8`（25.7 MB） | ❌ **无 int8/binary/float16 向量字段** → 只能 `FLOAT_VECTOR`（103 MB），或改用索引级 `HNSW_SQ` 压缩索引 |
| 增量缓存 | 自建 `vectors` 表 + 内容哈希比对 | ✅ **主键 = `content_hash` + `upsert`**，删除/更新天然支持（自建逻辑更少） |
| 元数据过滤 | numpy 布尔掩码（自己写 `_mode_ok`/`_version_ok`） | ✅ Milvus 标量过滤表达式，但**必须与 BM25 侧语义逐字一致**（`mode == "MySQL" or mode == ""`、`version` 前缀 LIKE → 用 `like`/范围表达式复现），且要有单测锁住两条路一致 |
| 原子重建 | 临时文件 + `os.replace` | 临时**目录**构建 → 关闭 client → 目录重命名/切换；注意文件锁 |
| artifact 缓存 | 单个 `.db` 文件 | 缓存**整个 data_dir**（WAL + Parquet + `.idx` + manifest） |
| 并发 | `RLock` 内只读，安全 | **每个 data_dir 单进程**（文件锁）、同 collection 写入需串行 → 与项目既有 `--workers 1` 约束一致，需在启动时显式校验并给出明确报错 |
| gitignore | `/backend/doc/*` 已覆盖 | 同样放在 `backend/doc/ob_wiki.milvus/`，规则已覆盖 |

### F.3 不要用的 Milvus 能力（重要）

1. **不要用 Milvus 的 BM25 取代 FTS5**（完整理由见 §3.5）：其 **BM25 IDF 统计是段内（segment-local）而非全局**，会改变排序口径；而我们现有的 FTS5 BM25 是四列权重 + 一系列手调常数（同义词奖励、导航惩罚），并有评测基线锁住。**Milvus 只承担稠密一路**，稀疏一路仍是 FTS5，融合仍用自己的 RRF（Milvus 内建 `RRFRanker` 只在两路都在 Milvus 内时可用）。
2. **不要依赖 Milvus 的 `TEXT_EMBEDDING` 函数**（当前仅支持 OpenAI，且需外网凭据）：嵌入由我们自己的 `EmbeddingClient` 计算后 `insert`，这样 `bge-m3`/内部端点/`fake`（测试）都能用。
3. **不要用 Milvus 的语义 rerank**（当前仅支持 Cohere）——rerank 属 P2a，由本方案的 **API rerank** 独立实现（§8.6），以便自由选择端点与模型。
4. **不要把本地 gRPC server 暴露到不可信网络**：无认证、无 RBAC、无 TLS。生产用进程内 `.db` 模式即可。

### F.4 为什么它反而值得选

1. **代码可平移**：`pymilvus` 客户端代码在 Lite / Standalone / Distributed / Zilliz Cloud 间基本不变 → §14.4 的 P4 迁移从「重写存储层」降级为「改 URI + 重建索引」。
2. **少写代码**：`upsert`（增量缓存）、标量过滤（mode/version）、`delete by filter`、`group-by`、迭代器都是现成的，省掉本方案 §5.2 里自建的比对/编码/矩阵加载逻辑。
3. **更少的内存驻留**：不必把 10 万级向量全量载入进程内存（虽然当前 25k 规模无所谓），`BRUTE_FORCE`/`FLAT` 在 Lite 上即可给出精确结果。
4. **仍然离线**：嵌入式、无外部服务，`mock` 演示的离线承诺不受影响（只是依赖体积变大）。

### F.5 代价与回退

- **新增依赖约 73 MB wheel（安装后更多）**，与项目「零依赖离线演示」的气质有张力；因此保留 `vector_backend: sqlite_numpy` 作为**零新增依赖的回退**（也是无 Milvus 环境下的单测后端）。
- **Beta 状态 + 3.x 与旧 v1 `.db` 格式不兼容** → 精确 pin 版本；向量是派生数据，最坏情况重新计算。
- 回退路径：`retrieval.provider=bm25`（不依赖任何向量后端）；或 `vector_backend` 切换。

### F.6 尚未验证（本机条件不足，需在目标环境做）

本机只有 **Python 3.9.6**（`milvus-lite` 要求 ≥3.10，项目要求 ≥3.13），因此**没有跑通端到端**，以下需在 CI/开发机（Python 3.13）实测：

1. 安装体积与 `pip install "pymilvus[milvus-lite]"` 的实际膨胀；
2. `MilvusClient(local .db)` 在 `--workers 1` 下的启动/关闭行为，以及 data_dir 的实际文件清单（确认能否目录级原子替换）；
3. 24k × 1024 维 `FLAT`/`BRUTE_FORCE` 的检索延迟（预期与 numpy 精确余弦同量级，3–8 ms）；
4. 标量过滤表达式对 `mode`/`version` 的等价性（与 FTS5 侧逐字对齐）；
5. `upsert` 增量的实际耗时（24k 块首次导入 vs 增量几十块）；
6. **稀疏 BM25 的 IDF 语义与稳定性**（段内 vs 全局、分数是否随 compaction 漂移）——这是 §3.5 第 3 条的判据，需写一个对照实验：同一批数据在多次写入/合并前后，对同一 query 比较稀疏检索的分数与排序。

### F.7 对 §15 的增量任务

| # | 任务 | 估时 |
| --- | --- | --- |
| T4a | `VectorIndex` 协议 + `MilvusLiteIndex` 实现（collection schema = `content_hash` 主键 + `vector` + `path/section/mode/version/kind` 标量字段；`upsert` 增量；临时目录构建） | 2.0 d |
| T4b | `SqliteNumpyIndex` 实现（回退/单测，保留 int8 与内容哈希缓存） | 1.0 d |
| T4c | 两后端一致性测试（同一批块 → 同 top-N；模式/版本过滤等价） | 0.5 d |

> §15 的合计已并入 T4a/T4b/T4c（双后端与一致性测试）与 P2a（R1–R3，API rerank）：**≈17.5 人日**；若只做 hybrid（跳过 rerank）为 **14.5 人日**。