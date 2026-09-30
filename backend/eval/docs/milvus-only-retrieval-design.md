# OceanBase DB Agent 检索重构方案（Milvus-only，v2）

> **本文取代 [`hybrid-retrieval-design.md`](hybrid-retrieval-design.md)（v1）。**
> v1 的方案是「保留 SQLite FTS5 稀疏一路 + 新增 Milvus 稠密一路」的双引擎混合检索；**v2 的决策是反过来：删除 FTS5，稀疏与稠密两路全部由 Milvus Lite 承担**。
> v1 保留作为**被否决方案的取舍记录**（尤其 §3.5「为什么 BM25 仍用 FTS5」——v2 已按决策放弃该方案的收益，代价在 §16 逐条列出）。
>
> **v2 的最大不同：结论不再是推论，而是真机实测。** 本文所有 Milvus 相关数字都来自本机跑通的探针（环境见 §3.1，原始输出见附录 B），不再依赖 wheel 元数据推测。

- 版本：v2
- 日期：2026-09-30
- 关联代码：[`backend/app/agent/doc_index.py`](../../app/agent/doc_index.py)、[`backend/app/agent/tools.py`](../../app/agent/tools.py)、[`backend/eval/run_eval.py`](../run_eval.py)
- 探针脚本与原始输出：**[`backend/eval/docs/probes/`](probes)**（5 个脚本 + 5 份 JSON 输出，已入库，见附录 B）

---

## 0. 决策摘要（TL;DR）

| 项 | 决策 |
| --- | --- |
| 引擎数量 | **一个**：Milvus Lite（嵌入式、无服务、目录式 data_dir） |
| 稀疏一路 | **Milvus 内建 BM25**（`FunctionType.BM25` + `SPARSE_INVERTED_INDEX`），中文分析器用 **`jieba`** |
| 稠密一路 | 同一集合内的 `FLOAT_VECTOR`（默认 `IVF_FLAT`，1024 维）+ 外部 Embedding API |
| 融合 | **自研 RRF（`k=60`）**，两路各取 `pool_k=50` 后融合（内建 `RRFRanker` 已验证可用，见 §17.1，当前不采用） |
| FTS5 | **彻底删除**：虚拟表、4 路召回、CJK 预分词、BM25 手调常数、`ob_wiki.index.db`（67.6 MB）全部移除 |
| SQLite 侧车 / `sqlite_numpy` | **彻底删除**（生产与单测都只用 Milvus Lite） |
| 语料陈旧检测 | 新的 `ob_meta` 集合存 `schema_version` / `corpus_fingerprint` / `embedding_model` / `dims`（取代原 `meta` 表） |
| Rerank（P2a） | **不变**：API rerank、默认 `auto`、先于稠密/混合落地（理由见 v1 附录 E，实测缺口比例 8:2） |
| 降级 | embedding 挂 → **只跑稀疏一路**（仍可用）；Milvus 打不开 → 检索整体不可用 + `retrieval_degraded`，返回空结果而非 500 |
| 预估工作量 | **≈17.0 人日** = T0 1.0 + T1 2.0 + 检索重构 11.0 + P2a 3.0；**不含 rerank = 14.0 人日**（见 §15） |
| 主要代价 | 无引擎级回退；依赖 +359 MB；data_dir +240 MB（3.5×）；稀疏一路 0.3 ms → 14 ms |

---

## 1. 背景与本次变更

### 1.1 现状（改造前）

`search_docs` → `DocIndex.search`（[doc_index.py:451](../../app/agent/doc_index.py#L451)）→ `_rank`（[doc_index.py:482](../../app/agent/doc_index.py#L482)），完全建立在 **SQLite FTS5** 上：

- external-content 虚拟表 + 四列 bm25 权重 `(title, keywords, section, body) = (10, 6, 4, 1)`；
- 单字 + 双字 CJK 预分词、27 组同义词、41 个停用词/疑问词清洗；
- 4 路召回（正文 / 导航页 / 扩展词 / 版本）与一组手调常数：`VERSION_MATCH_BONUS=8.0`、`AND_ALL_TERMS_BONUS=10.0`、`SYNONYM_PENALTY=10.0`、`NAVIGATION_SECTION_PENALTY=12.0`、`NAVIGATION_FILE_PENALTY=40.0`；
- 索引体积 67.6 MB、FTS 查询 P50 **0.3 ms**、`search` 端到端 P50 ≈ 98 ms（其中 `_fingerprint()` 全库 stat = **51.3 ms**）；
- 真语料基线：5146 篇 → 25077 块，命中率@1 60.00% / @5 82.00% / @10 94.00%，MRR@10 0.704。

**代码耦合度**：`doc_index.py` 全文 782 行，其中 **337 行（43%）与 FTS5/BM25 强绑定**：

| 区段 | 行 | 说明 |
| --- | --- | --- |
| 常量区（`_BM25_WEIGHTS`、奖励/惩罚、停用词） | 56 | 随 FTS5 一起废弃 |
| `_cjk_tokens` / `_query_tokens` | 31 | 改用 jieba，本侧删除 |
| `_build` 建 `chunks` 表 + FTS5 虚拟表 | 75 | 改为写入 Milvus |
| `search` | 30 | 改为两路 Milvus 查询 |
| `_rank` / `_match` / `_match_expanded` | 102 | BM25 公式与常数，整体废弃 |
| snippet / FTS5 查询等 | 43 | 改为从 Milvus 取 `text` |

### 1.2 为什么改

1. **决策**：只维护一个检索引擎。向量侧已确定用 Milvus Lite（嵌入式、无服务、API 可平移到 Standalone/Distributed/Zilliz）。
2. **实测发现**（见 §3.5）：v1 反对「BM25 也进 Milvus」的两条主要理由**在本轮实测中不成立或可接受**——IDF 未观察到段内漂移；`literal` 类查询在 jieba 下命中正确。
3. **代价明确且可测**：稀疏一路从 0.3 ms 变成 6–14 ms（绝对量仍很小）、依赖 +359 MB、data_dir 3.5×。**失去引擎级回退**是这次变更最主要的风险（§16-1）。

### 1.3 v1 → v2 关键差异

| 维度 | v1（双引擎） | v2（Milvus-only） |
| --- | --- | --- |
| 稀疏一路 | SQLite FTS5（自建分词 + 手调常数） | Milvus 内建 BM25（jieba 分词，服务端 IDF） |
| 分词 | 自写单字+双字预分词 | jieba（Milvus 服务端，且 **Lite 不支持 `run_analyzer` 内省**） |
| 词权重 | 四列权重 10/6/4/1 | 单 `text` 字段 + **前缀重复近似**（§8.2） |
| 同义词 | FTS5 召回路 | **查询扩展**（把同义词追加进 query 文本） |
| 导航惩罚 / 版本奖励 | `_rank` 内部 | **取回后置调整**（§8.4），常数保留 |
| 后端可替换 | `VectorIndex` 协议 + `sqlite_numpy` 回退 | 单后端，删除协议与回退（-1.5 人日） |
| 索引存储 | `ob_wiki.index.db` 67.6 MB | `ob_wiki.milvus/` ≈ 240 MB |
| 索引版本 | `SCHEMA_VERSION` + `meta` 表 | `schema_version` + `ob_meta` 集合 |
| 依赖 | 0 新增（stdlib sqlite3） | +359 MB（milvus-lite/faiss/pyarrow/grpcio/jieba…） |
| 回退 | `provider=bm25` 一键回退 | **无配置级回退**（见 §14.3） |
| CI 快通道 | 有（BM25 无依赖恒跑） | **无**（所有检索测试都需要 Milvus 依赖） |

---

## 2. 目标与非目标

### 2.1 目标

1. **单一引擎**：稀疏 + 稠密 + 元数据都在 Milvus Lite 内，删除 FTS5 与 SQLite 侧车。
2. **质量不劣化**：在扩样后的评测集上，`hybrid` 的 hit@5 ≥ 88%、MRR@10 ≥ 0.78（沿用 v1 §11.4 判据）；`sparse` 单独一路不得明显劣于改造前 FTS5 基线（阈值在 T1 重测后校准，见 §11.2）。
3. **`literal` 类查询可用**：错误码 / 版本号 / 配置项名在 jieba 分词下仍能命中（探针初步通过，见 §3.5）。
4. **可重复、可增量**：按 `content_hash` 判断是否需要重新 embedding；重复构建不重复付费。
5. **失败可降级、可观测**：embedding 不可用时稀疏一路仍可服务；Milvus 不可用时明确报降级而非 500。

### 2.2 非目标

- 不保留 FTS5 / SQLite 检索路径（**明确删除**，不是"默认关闭"）。
- 不做本地模型推理（embedding / rerank 都走 API）。
- 不迁移到 Milvus Standalone / Distributed / Zilliz（P4 再评估；本次只保证 API 可平移）。
- 不改 `read_doc` 行为、不改 agent 事件流与 UI、不改 `search_docs` 的入参 schema。

---

## 3. 真机验证结论（本方案的基础）

> 本节是本方案与 v1 最本质的区别：**v1 的 Milvus 结论是"读文档 + 查 wheel 元数据"推出来的，v2 是本机跑出来的。**

### 3.1 验证环境

| 项 | 值 |
| --- | --- |
| 解释器 | CPython **3.13.15**（uv 管理，`~/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none`） |
| OS / 架构 | macOS（Apple Silicon, arm64） |
| `pymilvus` | **3.0.2** |
| `milvus-lite` | **3.2.1** |
| 伴随依赖 | `faiss-cpu 1.15.1`、`pyarrow 25.0.1`、`grpcio 1.84.0`、`orjson 3.12.0`、`numpy 2.5.3`、`protobuf 7.36.2`、`pandas 3.0.6`、`jieba` |
| 安装方式 | `pip install "pymilvus[milvus-lite]"` + **`pip install jieba`（必须单独装，见 §3.2）** |

> ⚠️ **环境陷阱（已在本次踩到）**：DSH 自带的 Python 3.12 运行时开启了 macOS library validation，`pip` 装的未签名原生扩展（`orjson.so`）会 `dlopen` 失败。**必须用 uv / python-build-standalone 之类的解释器**，否则连 `import pymilvus` 都过不去。CI 上（Linux）无此问题。

### 3.2 能力矩阵（实测，非文档推断）

| 能力 | 结论 | 证据 / 备注 |
| --- | --- | --- |
| `MilvusClient("./x.db")` 嵌入式启动 | ✅ | data_dir 是**目录** |
| `SPARSE_INVERTED_INDEX` + 用户自建稀疏向量 | ✅ | 自建 `{term_id: weight}` 稀疏向量按 IP 精确命中（本次未采用，作为备选） |
| **内建 BM25（`FunctionType.BM25`）** | ✅ | **必须 `enable_analyzer=True`**，否则报 `BM25 function 'bm25' input field 'text' must have enable_analyzer=True` |
| 中文分析器 | ✅ **但类型名是 `jieba`** | `analyzer_params={"type": "chinese"}` → `unknown tokenizer type: 'chinese' (supported: 'standard', 'jieba')` |
| jieba 依赖 | ⚠️ **不随 `milvus-lite` 自动安装** | 未装时报 `JiebaAnalyzer requires the 'jieba' package. Install it with: pip install jieba` |
| 服务端分词内省（`run_analyzer`） | ❌ **Lite 未实现** | `StatusCode.UNIMPLEMENTED: Method not implemented!` → 分词是**黑盒**，只能用检索结果间接验证 |
| 标量过滤复刻 FTS5 语义 | ✅ | `mode == "MySQL" or mode == ""`、`version like "4.2.5%"`、两者 `and` 组合 + `INVERTED` 索引全部正确 |
| 稀疏 + 标量过滤组合查询 | ✅ | `search(..., anns_field="sparse", filter=...)` |
| `hybrid_search` + `RRFRanker(60)` | ✅ | 单次往返内完成两路 + 融合（本次不采用，见 §17.1） |
| `upsert`（按主键替换） | ✅ | 同主键 `upsert` 后检索到新文本 |
| `delete(ids=[...])` | ✅ | 行数正确下降 |
| `flush` + 索引完成 | ✅ | `flush` 后 `describe_index().state == "Finished"`、`pending_index_rows == 0` |
| 多进程打开同一 data_dir | ❌ **被文件锁拒绝** | 第二进程：`ConnectionConfigException: Open local milvus failed` → **必须 `--workers 1`** |
| fork 安全性 | ⚠️ | gRPC 打印 `Other threads are currently calling into gRPC, skipping fork() handlers` → 避免 `--reload` / preload / fork 型多进程 |

### 3.3 性能实测（25k 块规模，稳态）

语料形态与真语料对齐：每块 **1800 字中文正文**（3000 词表随机取词，避免病态 posting list）+ 1024 维稠密向量 + `mode`/`version` 标量 + jieba 分析器 + 内建 BM25。

| 操作 | 实测 | v1 假设 / FTS5 | 差异 |
| --- | --- | --- | --- |
| 构建 25k 块（insert + 索引） | **32.3 s** | FTS5 重建 5–8 s | 4–6× 慢，绝对量可接受 |
| 稀疏一路（中文口语 query） | **P50 14.0 ms**（max 15.1） | FTS5 **0.3 ms** | **≈45× 慢** |
| 稀疏一路（`ORA-00942` 字面 query） | **P50 6.1 ms** | — | 字面 query 更快 |
| 稠密一路（`IVF_FLAT`） | limit=10 **8.3 ms** / limit=50 **12.8 ms** | 估 3–8 ms | 接近 |
| 内建 `hybrid_search`+RRF（limit=10） | **P50 20.8 ms** | — | 单次往返 |
| 两路分查 + 自研 RRF（推算） | ≈ 14 + 13 + 开销 ≈ **30–35 ms** | — | 双次往返，仍远低于门禁 |

**稠密索引类型对比**（24k×1024，limit=50，稳态）：

| 索引 | P50（跨两轮实测） | 说明 |
| --- | --- | --- |
| `IVF_FLAT` | **10.1–10.4 ms**（两轮一致） | 本次选它（`nlist=128`），最稳定 |
| `FLAT` | 24.6–26.4 ms | 精确但更慢 |
| `HNSW` | 10.1–33.9 ms（**轮间波动大**） | 冷态/首次查询明显更慢；此处无优势 |
| `AUTOINDEX` | 32.8 ms（单轮） | Milvus 默认自动索引，不选 |

> HNSW/AUTOINDEX 的轮间差异来自"首次查询冷态"（同一脚本内先跑的组合偏慢）。选 `IVF_FLAT` 的原因正是它两轮都在 10 ms 附近，可预测。

> **测量陷阱（两次踩到）**：插入后**立刻**检索会得到 180–215 ms 的假值（索引/compaction 尚在后台）。必须 `flush` → 等 `pending_index_rows == 0` / `state == Finished` → 静置后再测。v1 附录 F.6-3 的"预期 3–8 ms"由此被修正为 **10–13 ms**。

### 3.4 容量与依赖实测

| 项 | 实测 | v1 估计 | 备注 |
| --- | --- | --- | --- |
| 依赖体积（venv） | **318 MB**（不含 jieba）→ **359 MB**（含） | 73 MB | **4.4×**；`faiss-cpu`+`pyarrow`+`grpcio` 是大头 |
| data_dir（25k 块，含正文+稀疏+稠密） | **218 MB**（构建后）→ **239 MB**（多次查询后） | 103 MB（仅稠密向量） | **≈8.7 KB/块**；FTS5 索引 67.6 MB 的 **3.5×** |
| data_dir 布局 | `collections/<name>/partitions/_default/{data,indexes}`、`wal/`、`schema.json` | 目录 | 24k 稠密实测 15 个文件 |
| 段切分 | 24k 稠密被切成 2×12000 段 | — | LSM 行为确认 |
| 构建后索引状态 | `flush` 后立即 `Finished` | — | 25k 无需长等待 |

### 3.5 对 v1 判断的修正（重要）

| v1 的说法 | 实测结论 |
| --- | --- |
| §3.5-1「BM25 零回归资格会丢失」 | **成立**：jieba 分词 + 服务端 IDF + 单字段，四列权重与手调常数无法逐位复现 → 必须重测基线（§11.2）。这是本方案**真实付出的代价**。 |
| §3.5-2「中文分词必然分叉」 | **成立但可接受**：分词确实不同，但 `literal` 实测命中正确——`ORA-00942` → 正确文档（score 4.412）、`00942`（子串）→ 同一文档（2.137）、`4.2.5 版本升级` → 正确文档（8.182）。 |
| §3.5-3「IDF 段内统计 → 分数随 compaction 漂移」 | **❌ 未复现**：追加 300 条后，同一 query 的 top-5 分数**逐位不变**（3.120938 ×5，`drifted: []`）。注：该批文档高度同构，属**弱证据**，真语料需在 T1 复测。 |
| §3.5-4「耦合检索可用与向量基础设施」 | **成立**：这是本次变更的核心代价，见 §16-1。 |
| §3.5-5「内建融合器用不上」 | **成立**：`RRFRanker` 可用，但为保留导航惩罚/版本奖励的前置调整，仍用自研 RRF（§8.5）。 |
| §3.5-6「当前规模收益为零」 | **修正**：收益是"单引擎"的维护性；代价是延迟与体积（上表）。 |
| 附录 F.6-3「FLAT 延迟 3–8 ms」 | **修正为 10–13 ms**（`IVF_FLAT`，25k×1024，稳态）。 |

**结论**：v1 反对该方案的理由中，**第 3 条被推翻**，第 2 条降级为"需回归验证的已知差异"，第 1、4、5 条仍然成立且已作为本方案的显式代价记录。因此本方案**不是"没有代价"，而是"代价已知、可测、可接受"**。

---

## 4. 总体架构

```
检索请求（search_docs）
   │
   ├─ 语料陈旧检测（T0 优化后的 fingerprint 缓存，目标 <5 ms）
   │     └─ 与 ob_meta 的 corpus_fingerprint 比对 → 不一致则提示重建
   │
   ├─ 查询预处理
   │     ├─ 同义词扩展（27 组词表保留，改为**追加到 query 文本**）
   │     └─ mode/version 消歧（沿用现有 _detect_mode/_detect_version）
   │
   ├─ 稀疏一路（Milvus 内建 BM25）        ┌──────────────────────────────┐
   │    data=[扩展后的 query 文本]        │  Milvus Lite（单一进程内）    │
   │    anns_field="sparse"          ────▶│  集合 ob_chunks              │
   │    filter=标量过滤, limit=pool_k     │   · text  (jieba 分析器)     │
   │                                      │   · sparse(BM25 Function)    │
   ├─ 稠密一路（Embedding API → 向量）    │   · vector(FLOAT_VECTOR)     │
   │   数据流同 v1（保持 embedder 抽象）   │   · path/section/mode/...    │
   │    data=[query 向量]            ────▶│  集合 ob_meta（kv）          │
   │    anns_field="vector"               └──────────────────────────────┘
   │    filter=标量过滤, limit=pool_k
   │
   ├─ 后置调优（每路内部，§8.4）
   │     score −= 导航惩罚(12/40)；score += 版本命中奖励(8)
   │
   ├─ 自研 RRF 融合（k=60，§8.5）→ 全局导航惩罚只扣一次
   │
   ├─ Rerank（P2a，API，可关；失败保留融合顺序）
   │
   └─ MAX_CHUNKS_PER_PATH 与 limit 截断（在 rerank 之后）
```

---

## 5. Milvus 集合设计

### 5.1 `ob_chunks`（主集合）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `pk` | `INT64`（主键） | `xxh64(f"{path}#{section}#{seq}")`——**用身份而非内容做键**，避免"两篇文档有完全相同的段落"被合并成一行。`seq` 是同一 `(path, section)` 内的第几块（0 起）：**必须带**，因为 `_split_chunks` 会把超 1800 字的同一小节切成多块（现有库里实测 1630 组 `(path, section)` 重复、单组最多 29 块），只按 `path#section` 做键会在 upsert 时互相覆盖 |
| `content_hash` | `VARCHAR(32)` | 嵌入文本的哈希；用于**判断是否需要重新 embedding**（不是主键） |
| `text` | `VARCHAR(8000)`, `enable_analyzer=True`, `analyzer_params={"type":"jieba"}` | 被 BM25 分析的文本：`{标题(前缀重复)} > {小节} | {正文}`；同时作为 snippet 来源 |
| `sparse` | `SPARSE_FLOAT_VECTOR` | **由 BM25 Function 自动生成**，不手工写入 |
| `vector` | `FLOAT_VECTOR(1024)` | 稠密向量（embedding 模型维度，需与 `ob_meta.dims` 一致） |
| `path` / `section` / `title` | `VARCHAR` | 结果回传与 `read_doc` 对齐 |
| `mode` / `version` | `VARCHAR(16)` | 标量过滤（复刻 `mode == ? or mode == ""`、`version like ?`） |
| `kind` | `VARCHAR(8)` | `doc` / `nav`；导航行照常写入（**零向量**），稠密一路恒 `filter kind == "doc"`（§8.4） |

Function 与索引声明：

```python
schema.add_field("text", DataType.VARCHAR, max_length=8000,
                 enable_analyzer=True, analyzer_params={"type": "jieba"})
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                             input_field_names=["text"], output_field_names=["sparse"]))

ip.add_index("sparse",  index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
ip.add_index("vector",  index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})
ip.add_index("mode",    index_type="INVERTED")
ip.add_index("version", index_type="INVERTED")
ip.add_index("kind",    index_type="INVERTED")
```

> **`max_length=8000` 的依据**：milvus-lite 3.2.1 实测 `VARCHAR` 的上限按**字符**计，不是按字节（probe7 第 2 节：`max_length=100` 能写入 100 个汉字 = 300 字节）。1800 字中文正文加标题/小节前缀远小于 8000 字符；配置里的 `max_text_bytes=8000` 守卫仍按字节算，比 schema 更保守，安全。

### 5.2 `ob_meta`（键值集合，取代原 SQLite `meta` 表）

| 键 | 值 |
| --- | --- |
| `schema_version` | 集合结构版本；不匹配 → 触发重建 |
| `corpus_fingerprint` | `files,size,newest_mtime_ns`（沿用现有 `_fingerprint()` 三元组） |
| `embedding_model` / `dims` | 模型或维度变化 → 全量重算 |
| `analyzer` | `jieba`（记录分词器；更换等于换检索口径，必须重测基线） |
| `built_at` | 构建时间（运维与排障） |

> 备选：把这几项写进 data_dir 旁的 `meta.json`。选集合是为了"单一引擎"的一致性，代价是多一个集合与一次 `query`。

### 5.3 索引与容量

| 项 | 值 |
| --- | --- |
| 稀疏索引 | `SPARSE_INVERTED_INDEX`，`metric_type="BM25"` |
| 稠密索引 | `IVF_FLAT` + `COSINE`，`nlist=128`（25k 重开后实测 P50 ≈27 ms） |
| 标量索引 | `mode`/`version`/`kind` 建 `INVERTED` |
| 磁盘 | **≈8.7 KB/块** → 25077 块 ≈ **220–240 MB**（构建后 218 MB，查询后涨到 239 MB，需持续观察 compaction 是否回落） |
| 构建 | 25k 块 **75 s**（Linux 2 vCPU，不含 embedding；embedding 是大头，见 §12） |

> **延迟测量口径（M6 门禁必须遵守）**：milvus-lite v3 是纯 Python 进程内实现，**在建索引的那个进程里测延迟会得到约 10x 的伪影**（同一目录 26000 行：构建进程内 sparse P50 779 ms / dense 599 ms，close 后在新进程 reopen 同一目录 → sparse 109 ms / dense 27 ms）。所以基线一律「**构建 → close → 新进程 reopen 再测**」，且测量期间机器上不能有别的重活（2 vCPU 上并发两个 Milvus 进程会把 P50 从 ~80 ms 推到 ~800 ms）。Linux 权威读数（26k 行）：sparse P50 85–110 ms（随查询词命中量浮动）、dense 27–29 ms，内建 `RRFRanker` hybrid ≈100–140 ms。
>
> 索引类型在 25k 规模上几乎不影响延迟：`IVF_FLAT`（nprobe 1/8/32 = 23.1/24.8/27.7 ms）与 `HNSW`（ef 16/64/200 = 23.6/24.5/25.8 ms）同档；HNSW 反而多 22 MB 磁盘与 12.8 s 构建时间，**保持 `IVF_FLAT`**。

### 5.4 data_dir 与原子重建

- data_dir：`backend/doc/ob_wiki.milvus/`（**目录**，`.gitignore` 的 `/backend/doc/*` 已覆盖）。
- 原子重建：临时目录构建 → 关闭 client → **目录级重命名/切换**（Milvus 是 LSM，不能像单文件那样 `os.replace` 一个文件；必须整目录切换）。
- **单进程约束**：同一 data_dir 只能被一个进程打开（实测文件锁）。启动时显式校验并给出明确报错；`uvicorn --workers 1`，禁 `--reload`/preload（gRPC + fork 警告，见 §3.2）。

---

## 6. 配置设计

```yaml
retrieval:
  milvus_path: backend/doc/ob_wiki.milvus   # data_dir（目录）
  collection: ob_chunks
  meta_collection: ob_meta
  dense_index: IVF_FLAT                     # IVF_FLAT | HNSW | FLAT
  nlist: 128
  pool_k: 50                                # 两路各自候选池，与 limit 解耦
  rrf_k: 60
  weight_sparse: 1.0                        # RRF 权重（默认等权）
  weight_dense: 1.0
  dense_top_n: 50
  query_cache_size: 512
  query_timeout_seconds: 5
  nav_section_penalty: 12.0                 # 保留的调优常数（后置调整）
  nav_file_penalty: 40.0
  version_match_bonus: 8.0
  # 已删除：provider(bm25|hybrid)、vector_backend、quantization、
  #         max_vectors_in_memory、各 dense 预过滤开关（这些属于 FTS5/双后端时代）

embedding:
  base_url: ""        # 必填（非空校验）
  api_key: ""
  model: ""           # 必填；建议 bge-m3 或 text-embedding-3-large
  dims: 1024
  batch_size: 64
  concurrency: 4
  timeout_seconds: 30

rerank:               # P2a，与 v1 一致
  mode: auto          # auto | off | api
  base_url: ""
  api_key: ""
  model: ""           # api 且为空 → 启动失败
  top_n: 30
  timeout_seconds: 2
  max_passage_chars: 500
```

环境变量沿用 `_env_nonempty`（[config.py:161](../../app/config.py#L161)）语义：**空字符串不覆盖 YAML 非空值**；命名规则为 `RETRIEVAL_*` / `EMBEDDING_*` / `RERANK_*`。

**校验规则（fail fast）**

| 条件 | 行为 |
| --- | --- |
| `embedding.model` 为空 | 允许启动，但**只能跑稀疏一路**（`/api/health` 标记 `dense_disabled`） |
| `rerank.mode: api` 且 `rerank.model` / `base_url` 为空 | **启动失败** |
| `embedding.model` 非空但 `base_url` 为空 | **启动失败** |
| `milvus_path` 不存在 | 启动时提示"需先建索引"，检索返回空 + `retrieval_degraded` |
| data_dir 被其他进程占用 | **启动失败**，报错写明 `--workers 1` 约束 |
| `ob_meta.dims` 与 `embedding.dims` 不一致 | 启动失败，提示需 `--rebuild-vectors` |

---

## 7. 索引构建与增量

```
python -m app.agent.milvus_index --rebuild           # 全量重建（临时目录 → 原子切换）
python -m app.agent.milvus_index --rebuild-vectors   # 增量补向量（默认）
python -m app.agent.milvus_index --stats             # rows / coverage / dims / model / size
python -m app.agent.milvus_index --verify            # 两路可查 + meta 一致性
```

流程：

1. 扫描 `backend/doc/ob_wiki` → 解析 frontmatter → `_split_chunks`（**沿用现有实现**，39 行不动）。
2. 生成 `pk = xxh64(path#section#seq)` 与 `content_hash`（基于最终 `text`）；`seq` 为同一小节内的块序号（见 §5.1）。
3. **增量判定**：从 `ob_chunks` 查回现有 `pk → content_hash`，只对「新增」或「哈希变化」的块调用 embedding；未变的块**跳过费用**。
4. 组装行：`text` = `{标题前缀重复} > {小节} | {正文}`（§8.2 的权重复刻），`text` 与 `vector` 一起 `upsert`。
5. **剪枝**：删除本次不再存在的 `pk`（文件被删/小节改名）：`delete(filter="pk in [...]")` 分批执行。
6. 写入 `ob_meta`：`schema_version` / `corpus_fingerprint` / `embedding_model` / `dims` / `analyzer` / `built_at`。
7. 原子切换临时目录。

> **导航文件（`kind='nav'`）写入 `ob_chunks` 但不嵌入**（M3 决议，取代上一版「不写入」的说法）：导航行照常参与稀疏一路，稠密一路恒 `filter kind == "doc"`，因此 `include_index=true` 时导航页仍可出现（走 BM25），而稠密一路永远不会把导航页顶到前面。**为什么必须写进去**：`_is_navigation_section` 的 −40 惩罚与「导航页永不 top1」硬门禁（[tests/test_retrieval_eval.py:71](../../../tests/test_retrieval_eval.py#L71)）都靠这一行的存在；而 `FLOAT_VECTOR` 不可空（probe7 第 1 节），所以导航行的向量写零向量（COSINE 下 distance=0，只要不加过滤也不会盖过真正相关的 doc 行）。**代价**：导航行多占一份 `text` 存储。

---

## 8. 检索流程

### 8.1 对外签名（与 v1 一致，工具层无感）

```python
def search(
    self,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    mode: str = "",
    version: str = "",
    include_index: bool = False,
    retriever: str = "",   # "" = 配置默认; "sparse" | "dense" | "hybrid"
) -> list[dict[str, Any]]:
```

- 工具层 `search_docs`（[tools.py:449](../../app/agent/tools.py#L449)）**入参 schema 不变**；`retriever` 仅供评测/单测显式传参。
- 返回**只增不改**：既有 `{path, section, title, mode, version, score, snippet}` 保留，新增诊断字段 `sources` / `sparse_rank` / `dense_rank`。
- 工具层返回键名仍为 `{query, hits, hit_count}`。

### 8.2 稀疏一路（内建 BM25 + jieba）

```python
def _sparse_rank(self, query, *, mode, version, include_index, pool_k):
    q_text = _expand_synonyms(query)                    # 27 组同义词表 → 追加词面
    flt = _scalar_filter(mode=mode, version=version, include_index=include_index)
    hits = self._client.search(self._collection, data=[q_text], anns_field="sparse",
                               filter=flt, limit=pool_k,
                               output_fields=["path", "section", "title", "kind"])
    return hits[0]
```

**四列权重的复刻（近似）**：BM25 只分析一个字段，因此把标题与关键词**重复写入 `text` 前缀**来近似加权：

```
text = (标题 ×3) + " " + (关键词 ×2) + " " + 小节 + " | " + 正文
```

> 这是**近似而非等价**：BM25 的 `tf` 饱和（`k1`）会让重复 3 次的实际权重远小于 3×。**必须用评测确认**，`literal` 门禁是硬约束（§11.4）。若近似不足，备选是"自建稀疏向量"（探针已验证可用，见 §3.2），由我们在客户端算 `tf`/`idf` 以精确复刻权重——**代价是自行维护 df 统计**。

**同义词**：原 FTS5 里同义词参与召回路径与惩罚计算，v2 改为**查询扩展**（把同义词词面追加进 `q_text`），`SYNONYM_PENALTY` 常数随之不再适用（删除）。

### 8.3 稠密一路

```python
def _dense_rank(self, query, *, mode, version, pool_k):
    q = self._embed_query(query)                       # LRU 缓存；失败抛 VectorUnavailable
    flt = _scalar_filter(mode=mode, version=version, include_index=False)  # 稠密永不返回 nav
    hits = self._client.search(self._collection, data=[q], anns_field="vector",
                               filter=flt, limit=pool_k,
                               output_fields=["path", "section", "title"])
    return hits[0]
```

- `mode`/`version` 过滤表达式**必须与稀疏一路逐字一致**（同一 `_scalar_filter`），并有单测锁住。
- 过滤后为空不做"回退不过滤"，只让 RRF 用另一路。

### 8.4 后置调优（保留的调优资产）

FTS5 删除后，以下常数没有落点了，改为**取回结果后调整分数**（在融合之前、每路内部）：

| 常数 | 原作用 | v2 落点 |
| --- | --- | --- |
| `NAVIGATION_SECTION_PENALTY=12.0` | 导航小节降权 | 两路内部 `score -= 12`（判定用 `_is_navigation_section`，**该函数保留**） |
| `NAVIGATION_FILE_PENALTY=40.0` | 导航文件降权 | 稀疏一路内部 `score -= 40`（`kind='nav'`）；稠密一路不返回 nav，天然免疫 |
| `VERSION_MATCH_BONUS=8.0` | 版本命中奖励 | 两路内部 `score += 8`（`_version_ok` 判定，**保留**） |
| `AND_ALL_TERMS_BONUS=10.0` | 全词命中奖励 | **删除**：jieba 下"全词"概念不同，且 BM25 已按词项匹配打分 |
| `SYNONYM_PENALTY=10.0` | 同义词惩罚 | **删除**（同义词改为查询扩展，不再需要惩罚） |
| `_BM25_WEIGHTS=(10,6,4,1)` | 四列权重 | **近似复刻**（§8.2 前缀重复） |

- **导航惩罚全局只扣一次**：两路内部各扣一次之后，融合结果上**不再重复扣**（用与 v1 相同的 `nav_penalty_done` 标志）。
- 注意：这里调整的是**分数**，而 RRF 只用**排名**。因此调优发生在"进入 RRF 之前"的每路排序上——两路各自 `score` 调整后重新排序，再取 rank。语义与 v1 一致。

### 8.5 融合（自研 RRF）

```python
fused = _rrf(sparse_hits, dense_hits, k=60, w=(weight_sparse, weight_dense))
```

- 只用排名，规避 BM25 分数与 cosine 量纲不可比的问题（v1 附录 A 的理由仍然成立）。
- **不使用** Milvus 内建 `RRFRanker`：它无法在融合前做 §8.4 的每路分数调整（实测可用，作为备选见 §17.1）。
- 每路各取 `pool_k=50`，与输出 `limit` 解耦（修 v1 记录的"排序随 `limit` 漂移"问题）。

### 8.6 Rerank（P2a，与 v1 一致）

| 项 | 方案 |
| --- | --- |
| 形态 | **API rerank**（默认 `auto`，不做本地 ONNX/GPU） |
| 输入 | 融合后的前 `rerank.top_n`（默认 30） |
| 打分文本 | `标题 > 小节` + 300–500 字摘要片段 |
| 超时/失败 | `rerank.timeout_seconds=2` → **保留融合顺序**，`/api/health` 标 `rerank_degraded` |
| 每篇上限与截断 | `MAX_CHUNKS_PER_PATH=2` 与 `limit` 截断**在 rerank 之后** |

---

## 9. 删除清单与新增清单

| 文件 | 动作 | 说明 |
| --- | --- | --- |
| [`app/agent/doc_index.py`](../../app/agent/doc_index.py) | **删除 337 行 / 改写约 120 行** | 删除 `_cjk_tokens`、`_query_tokens`、`_BM25_WEIGHTS`、`_STOPWORDS`、`_rank`、`_match`、`_match_expanded`、FTS5 建表、snippet SQL；保留 `_split_chunks`、`_parse_frontmatter`、`_detect_mode`/`_detect_version`、`_is_navigation_section`、`_excerpt`、`read`、`to_wiki_path`、`_fingerprint` |
| `app/agent/milvus_index.py`（新） | 新增 | Milvus client 封装、schema/function/index 声明、`ob_meta` 读写、原子重建、单进程校验 |
| `app/agent/retrieval.py`（新，可选拆分） | 新增 | `_sparse_rank`、`_dense_rank`、`_rrf`、后置调优 |
| `app/agent/rerank.py`（新） | 新增（P2a） | `Reranker` 协议 + `ApiReranker` |
| `app/agent/embedding.py`（新） | 新增 | `EmbeddingClient`（OpenAI 兼容 `/embeddings`） |
| [`app/agent/tools.py`](../../app/agent/tools.py) | 微改 | 异常分支 `VectorUnavailable` / `RerankUnavailable` → `hint`；**入参 schema 不变** |
| `app/config.py` | 改 | 新增 `RetrievalConfig`（当前**没有**任何检索配置类，见 §6） |
| [`backend/requirements.txt`](../../requirements.txt) | 改 | 新增 `pymilvus`、`milvus-lite`、`jieba`（+ 传递依赖 `faiss-cpu`、`pyarrow`、`grpcio`、`protobuf`、`pandas`）；`numpy`、`orjson`、`requests` 已在 |
| `backend/doc/ob_wiki.index.db` | **删除** | 67.6 MB，FTS5 索引不再需要 |
| `tests/test_doc_index.py` | 重写 | 现有测试大量针对 FTS5 机制（分词、降权、切片 SQL）；改为针对 Milvus 行为 + 契约 |
| `tests/test_retrieval_eval.py` | 改 | 真语料评测走 Milvus |
| [`backend/eval/run_eval.py`](../run_eval.py) | 改 | `--retriever sparse\|dense\|hybrid`、`--min-hit1`；去掉 `--dense-top-n` 等双后端参数 |

**代码层面最大的一处简化**：`doc_index.py` 里的 `sqlite3` import、`_connect`、`ensure`、`_is_current`（用 `meta` 表）全部改为 Milvus + `ob_meta`；**项目不再使用 SQLite 做检索**（其余功能未受影响，`grep` 确认只有这 4 个文件依赖 `doc_index`）。

---

## 10. 降级与可观测

**不变量**：任何检索相关故障都**不得**返回 500、不得让 agent 崩溃。

| 故障 | 行为 | 观测 |
| --- | --- | --- |
| embedding 端点超时/不可用 | **只跑稀疏一路**（BM25 仍在 Milvus 内，检索可用） | `dense_degraded: true` |
| embedding 未配置 | 与上同（常态降级，不算故障） | `dense_disabled: true` |
| rerank 端点超时/不可达 | 保留融合顺序 | `rerank_degraded: true` |
| **Milvus data_dir 打不开 / 被锁** | 检索返回空列表 + 说明性 `hint`（"检索索引不可用"） | `retrieval_degraded: true`（**新单点**） |
| `ob_meta` 与语料指纹不一致 | 正常检索，但提示需重建 | `index_stale: true` |

> `/api/health` 只增字段，前端 `HealthBadge` 忽略未知字段。

---

## 11. 评测与验收

### 11.1 评测器改造

```bash
python backend/eval/run_eval.py --retriever sparse|dense|hybrid \
       --rerank auto|off|api --min-recall 0.80 --min-mrr 0.68 --min-hit1 0.65
```

报告新增：`by_source`（来自哪一路）、`rerank_moved`（前移/后移用例）、`by_tag.*`。

### 11.2 基线必须重建（v2 的核心验收工作）

FTS5 基线（hit@1 60.00% / @5 82.00% / MRR@10 0.704）**不再可比**——分词器、IDF、字段权重全变了。因此：

1. **T1 先在 Milvus 上重测 `sparse` 单独一路**，得到 v2 的新基线，写入 [`backend/eval/README.md`](../README.md)。
2. **接受判据**（需 T1 后用实测校准）：
   - `sparse` 一路：hit@5 **不低于改造前 82% 的 -2pp**（即 ≥80%）；`literal` 类不得劣化超过 1 条；
   - `hybrid`：hit@5 ≥ 88%、MRR@10 ≥ 0.78（沿用 v1 目标）；
   - `rerank`：hit@1 相对新基线 **≥ +5pp**（`--min-hit1`）。
3. **jieba 质量评估**（T1 必做）：把 49 条用例按 `literal` 切出来，逐条比对"改造前命中 / v2 命中"，**任何一条 literal 退化都要单独解释**。

### 11.3 CI 通道（因无 FTS5 而重写）

| 通道 | 触发 | 内容 | 依赖 |
| --- | --- | --- | --- |
| **A（恒跑）** | 每个 PR | `--retriever sparse --rerank off`：**新基线**零回归 + `by_tag.literal` 不劣化 | 需装 Milvus 依赖（无快通道了） |
| **B（恒跑）** | 每个 PR | `--retriever hybrid --rerank off`：质量门禁 + 向量 artifact 缓存 | 同上 + embedding key 或缓存 artifact |
| **C（nightly / 有密钥）** | 定时 | `--retriever hybrid --rerank api --min-hit1 0.65` | rerank key |

> 通道 A 不再"快、无密钥"：Milvus 依赖安装（≈359 MB）是新增 CI 成本。可选优化：缓存 `pip` wheel + 缓存 data_dir artifact。

### 11.4 七道门禁

1. `hybrid` hit@5 ≥ 88%、MRR@10 ≥ 0.78（T1 校准）；
2. `sparse` 不劣于新基线；
3. `by_tag.literal` 不劣化（相对劣化 ≤1 条）；
4. 导航页不得被顶到正文之前（`test_navigation_pages_never_top_body_answers` 重写后仍须通过）；
5. `bm25` 路径的 `no-500` 降级断言（embedding 注入超时）；
6. 延迟门禁（§12.2）；
7. rerank `--min-hit1 0.65`（通道 C）。

---

## 12. 性能与成本

### 12.1 构建成本

| 项 | 值 |
| --- | --- |
| 块数 | 25077（23995 doc 嵌入；1082 nav 不嵌入） |
| embedding 请求 | 23995 / batch 64 ≈ 375 次（并发 4） |
| embedding token | 中文约 6–8M token（`text` 现已含标题前缀重复，比 v1 略增） |
| 时间 | 自建端点约 10–30 分钟；托管 API 约 3–10 分钟；**Milvus 写入本身仅 32.3 s** |
| 增量 | 按 `content_hash` 跳过未变块；典型改一篇文档 → 个位数请求 |

### 12.2 查询延迟（25k 规模实测）

| 阶段 | 实测 P50 | 说明 |
| --- | --- | --- |
| 稀疏一路 | 6–14 ms | 中文 query 14 ms；字面 query 6 ms |
| 稠密一路 | 8–13 ms | limit=10 → 8.3 ms；limit=50 → 12.8 ms |
| 自研 RRF 融合 | <1 ms | 纯 Python |
| 语料指纹（现状） | **51.3 ms** | **最大单项**，T0 优化目标 <5 ms |
| rerank（API） | 100–500 ms | 两点网络调用 |
| **合计（rerank off）** | **≈70–100 ms** | 含指纹 |
| **合计（rerank api）** | **≈170–600 ms** | |

**门禁（比 v1 可收紧）**

- `sparse + rerank off`：P50 ≤ 50 ms、P95 ≤ 150 ms（原 10 ms 门禁随 FTS5 一起作废）；
- `hybrid + rerank off`：P50 ≤ 150 ms、P95 ≤ 300 ms；
- `hybrid + rerank api`：P50 ≤ 800 ms、P95 ≤ 1.8 s。

### 12.3 容量与成本

| 项 | v1（FTS5+侧车） | v2（Milvus-only） | 变化 |
| --- | --- | --- | --- |
| 索引/数据目录 | 67.6 MB | **218–239 MB** | 3.3–3.5× |
| Python 依赖 | 0 新增 | **+359 MB**（venv） | 显著 |
| 内存 | numpy 矩阵（可选 int8） | Milvus Lite 托管（无 int8） | 由引擎管理 |

---

## 13. 安全

- `text`（文档正文，含内部产品文档）现在落在 **Milvus data_dir** 而非 SQLite 文件——`backend/doc/*` 已 gitignore，需在部署文档中明确"该目录含文档原文"。
- embedding / rerank 会把**查询文本与文档片段**发出外网（与 v1 相同），需在配置说明与 README 中披露；支持指向内部端点。
- Milvus 本地 gRPC server 模式**无认证/无 RBAC/无 TLS**，不得暴露到不可信网络；本方案只用进程内模式。
- data_dir 含原文但无凭据；备份/清理策略与语料一致。

---

## 14. 迁移与回滚

### 14.1 迁移顺序

1. **T0 先落**（指纹缓存，1.0 d，与引擎无关，零风险）；
2. **T1 扩样 + Milvus 基线重测**（2.0 d）——**必须先完成**，否则后面的质量判断没有基准；
3. 建 Milvus 基建与索引（§9），**保留 FTS5 代码在分支上**，用 `--retriever sparse|hybrid` 做对比评测；
4. 达标后切换默认路径，**删除 FTS5 代码与 `ob_wiki.index.db`**；
5. 删除 `sqlite_numpy` / `VectorIndex` 协议（已无第二后端）。

### 14.2 灰度与验证

- 上线前后各跑一遍全量评测，保存 `--json` 报告 diff（`rescued` / `broken` 逐条解释）。
- 生产观察：`/api/health` 的 `retrieval_degraded` / `dense_degraded` / `index_stale` 三个标志。

### 14.3 回滚（**这是 v2 最大的弱点，必须写清楚**）

| 场景 | 回滚手段 | 代价 |
| --- | --- | --- |
| 质量不达标（发现于灰度） | 回滚**代码/镜像**到 FTS5 版本 | 需要保留旧镜像；无配置级开关 |
| 质量不达标（已上线） | 回滚镜像 + 旧 `ob_wiki.index.db` | 索引文件需保留到确认稳定为止 |
| Milvus Lite 踩坑（Beta/格式） | pin 版本回退；最坏重建 data_dir | 向量是派生数据，可重建 |
| data_dir 损坏 | `--rebuild` 重建 | 需重新 embedding（费用） |

> **明确取舍**：v1 的 `RETRIEVAL_PROVIDER=bm25` 一键回退**在 v2 不存在**。建议在上线后**至少保留一个发布周期的 v1 镜像与 `ob_wiki.index.db`**（67.6 MB，成本极低）作为唯一退路——虽然代码里不再有 FTS5，但镜像里有。

---

## 15. 里程碑与任务拆解

| 阶段 | # | 任务 | 产出 | 估时 |
| --- | --- | --- | --- | --- |
| 性能前置 | T0 | 指纹缓存（`_fingerprint` 51.3 ms → <5 ms） | 与引擎无关的性能收益 | 1.0 d |
| 评测前置 | T1 | 评测集扩样 150–200 条 + **Milvus 新基线** + jieba/literal 质量评估 | `retrieval_cases.jsonl`、新基线写入 README | 2.0 d |
| | T2 | Milvus 基建：client 封装、schema/function/index、`ob_meta`、原子目录重建、单进程校验、依赖接入 | 可建可查 | 1.5 d |
| | T3 | 索引构建与增量：`pk`/`content_hash`、upsert、剪枝、embedding 跳过、`--stats/--verify` | 增量构建可用 | 2.0 d |
| **检索重构（P2b）** | T4 | 检索重构：`_sparse_rank`/`_dense_rank`/后置调优/自研 RRF + 签名与诊断字段 | 三模式可评 | 2.5 d |
| | T5 | **删除 FTS5**（337 行）+ 重写 `test_doc_index.py` + 4 个调用方改造 | 单一引擎 | 2.0 d |
| | T6 | 评测器（`--retriever`/`--min-hit1`）+ CI 三通道 + artifact 缓存 | CI 绿 | 1.5 d |
| | T7 | 配置/示例/README/运维手册/回滚演练 | 可交付 | 1.5 d |
| | | **小计（P2b）** | | **11.0 d** |
| **Rerank（P2a，先行）** | R1 | 固定候选池 + `rerank.py`（协议 + `ApiReranker`） | `hybrid + rerank api` 可跑 | 1.0 d |
| | R2 | `search` 接入 rerank + 每篇上限后移 + 降级路径 | 质量提升可测 | 1.0 d |
| | R3 | rerank 评测与门禁（`--min-hit1`、`by_tag.literal`） | 上线判据 | 1.0 d |
| | | **小计（P2a）** | | **3.0 d** |
| | | **合计** | | **≈17.0 d**（不含 rerank **14.0 d**） |

**执行顺序**：T0 → T1 → **P2a（R1–R3）** → P2b（T2–T7）。理由见 v1 附录 E（实测缺口比例 8:2，重排杠杆大于扩召回），该结论与引擎选择无关，**在 v2 中依然成立**。

> **与 v1 的工作量差异**：v1 17.5 d → v2 17.0 d。删掉 `sqlite_numpy` 与双后端一致性测试省 1.5 d；新增 FTS5 删除（337 行）与测试重写多花约 1.0 d。

---

## 16. 风险登记

| # | 风险 | 影响 | 概率 | 缓解 |
| --- | --- | --- | --- | --- |
| 1 | **无引擎级回退**：Milvus 打不开 = 检索完全不可用 | 检索整体中断 | 中 | 保留一个周期的 v1 镜像（§14.3）；`--verify` 健康检查；启动即校验 data_dir 可打开；`retrieval_degraded` 明确暴露 |
| 2 | **Milvus Lite 处于 Beta**，3.x 与旧 v1 格式不兼容 | 升级即需重建 | 中 | 精确 pin 版本；向量是派生数据，可重建 |
| 3 | **jieba 分词改变检索口径**，`literal` 类退化 | 错误码/版本号查询变差 | 中 | 探针已初步通过（§3.5）；T1 用 `by_tag.literal` 逐条比对；备选"自建稀疏向量"精确复刻权重 |
| 4 | **依赖 +359 MB、data_dir 240 MB** | 与"轻量离线演示"气质冲突；镜像/CI 变重 | 高 | 缓存 wheel 与 data_dir artifact；镜像分层；README 明确体积 |
| 5 | **单进程文件锁 + gRPC fork 警告** | 多 worker / `--reload` 直接失败 | 中 | 启动校验 + 明确报错；文档写明 `--workers 1`、禁 preload/reload |
| 6 | 稀疏一路延迟 0.3 ms → 6–14 ms | 端到端变慢 | 低 | 绝对量仍 <20 ms；指纹（51.3 ms）才是瓶颈，由 T0 解决 |
| 7 | 四列权重只能"前缀重复"近似 | BM25 排序质量不及 FTS5 调优结果 | 中 | 评测校准；`literal` 门禁兜底；备选自建稀疏向量 |
| 8 | IDF 段内统计（探针未复现，但证据弱） | 分数随写入漂移 → 门禁不稳 | 低 | T1 用真语料复测（追加写入前后比对分数）；若有漂移改为定期重建 |
| 9 | `run_analyzer` 不可用 → 分词黑盒 | 分词问题只能靠检索结果反推 | 低 | 建立"分词行为快照"测试：固定 query 集合的命中结果作为回归基线 |
| 10 | data_dir 查询后从 218 → 239 MB | 磁盘缓慢增长 | 低 | 观察 compaction 是否回落；必要时定期 `--rebuild` |
| 11 | 所有检索测试都需 Milvus 依赖 | CI 变慢（无快通道） | 中 | wheel 缓存 + data_dir artifact；单测中复用同一 data_dir 夹具 |

---

## 17. 待决问题

### 17.1 已决

| 问题 | 决策 |
| --- | --- |
| 单一引擎还是双引擎 | **单一引擎（Milvus Lite）**；FTS5 与 SQLite 侧车全部删除 |
| 稀疏一路实现 | **Milvus 内建 BM25 + jieba** |
| 第二后端（`sqlite_numpy`） | **删除**，生产与单测都只用 Milvus Lite |
| 中文分析器类型名 | **`jieba`**（不是 `chinese`）；且必须显式 `pip install jieba` |
| `enable_analyzer` | **必须为 True**，否则无法建 BM25 Function |
| 稠密索引类型 | **`IVF_FLAT` + COSINE（nlist=128）** |
| rerank 形态 | **API、默认 `auto`、不做本地 ONNX**（沿用 v1） |
| 执行顺序 | T0 → T1 → P2a → P2b（沿用 v1） |

### 17.2 待定

1. **融合用自研 RRF 还是内建 `RRFRanker`**：实测内建 hybrid 单次往返 P50 20.8 ms，自研两路分查约 30–35 ms（多一次往返，但可在每路内部做 §8.4 的分数调整）。**推荐自研**，除非延迟成为问题。
2. **四列权重的近似方式**：前缀重复次数（标题 ×3？×5？）需 T1 评测校准；若近似不足，是否改用"自建稀疏向量"（已验证可用）。
3. **27 组同义词表**的处置：改为查询扩展后，是否需要按词表分组给不同权重。
4. **`ob_meta` 还是 data_dir 旁 `meta.json`**：前者一致性好，后者少一个集合、读取更快。
5. **是否保留 FTS5 代码在分支/tag**：影响回滚手段（§14.3）。
6. **CI 是否安装完整 Milvus 依赖**，还是只用缓存的 data_dir artifact 跑检索测试。
7. **`milvus-lite` 版本 pin 策略**：3.x Beta，是否 pin 到 patch 版本。
8. **embedding 端点与模型**（`bge-m3` vs `text-embedding-3-large`）、**维度是否固定 1024**（沿 v1 未决项）。
9. **data_dir 是否需要定期重建**以控制体积（观察 compaction 后再定）。

---

## 附录 A：术语

| 术语 | 含义 |
| --- | --- |
| BM25 Function | Milvus 服务端函数：写入 `text` 时自动生成稀疏向量，检索时自动分析 query（本次的中文分支为 jieba） |
| `SPARSE_INVERTED_INDEX` | Milvus 的稀疏倒排索引，`metric_type="BM25"` 时承载 BM25 打分 |
| `pool_k` | 每路候选池大小（固定 50），与输出 `limit` 解耦 |
| RRF | Reciprocal Rank Fusion，只用排名融合两路 |
| data_dir | Milvus Lite 的数据目录（WAL + Parquet + 索引 + manifest），**不是单文件** |
| `ob_meta` | 存 schema 版本、语料指纹、模型与维度的键值集合 |

## 附录 B：探针脚本与原始输出

脚本与输出都在 [`backend/eval/docs/probes/`](probes)（**已入库**，评审可直接核对数字）：

| 脚本 | 输出 | 覆盖内容 |
| --- | --- | --- |
| [`probe1_basics.py`](probes/probe1_basics.py) | [`probe1.out.json`](probes/probe1.out.json) | 启动、24k 稠密、用户稀疏向量、BM25（缺 `enable_analyzer` → 失败记录）、标量过滤、data_dir 布局、多进程锁 |
| [`probe2_tokenizer_index.py`](probes/probe2_tokenizer_index.py) | [`probe2.out.json`](probes/probe2.out.json) | `chinese` 类型名错误、`run_analyzer` 未实现、稠密索引类型对比 |
| [`probe3_jieba_dep.py`](probes/probe3_jieba_dep.py) | [`probe3.out.json`](probes/probe3.out.json) | jieba 依赖缺失（`JiebaAnalyzer requires the 'jieba' package`）、干净稠密延迟 |
| [`probe4_bm25.py`](probes/probe4_bm25.py) | [`probe4.out.json`](probes/probe4.out.json) | **内建 BM25 全项**：`literal` 命中、24k BM25 延迟、IDF 漂移、标量过滤、内建 hybrid、upsert/delete |
| [`probe5_25k_steady.py`](probes/probe5_25k_steady.py) | [`probe5.out.json`](probes/probe5.out.json) | **25k 真实形态**：构建 32.3 s、data_dir 218→239 MB、稳态稀疏/稠密/hybrid 延迟 |

复现方式：

```bash
# 必须用 uv / python-build-standalone 解释器（见 §3.1 的 library validation 陷阱）
PY=~/.local/share/uv/python/cpython-3.13.15-macos-aarch64-none/bin/python3
$PY -m venv venv && ./venv/bin/python -m pip install "pymilvus[milvus-lite]" jieba
./venv/bin/python backend/eval/docs/probes/probe5_25k_steady.py     # 约 3–6 分钟，需要 ~1 GB 磁盘
```

> 探针会在 `/tmp/milvus_spike/` 建 data_dir，可反复执行（脚本内先删旧目录）。

**关键原始值**（probe5，25k 块）：

```json
{
 "insert_25k_s": 32.3,
 "last_status": {"vec_pending": 0, "vec_state": "Finished",
                 "sparse_pending": 0, "sparse_state": "Finished", "row_count": 25000},
 "load_state": "{'state': <LoadState: Loaded>}",
 "data_dir_mb": 218.3,
 "bytes_per_chunk": 8732.0,
 "steady_state_latency_ms": {
   "sparse:锁等待的 SQL 怎么看": {"p50": 14.0, "max": 15.1},
   "sparse:ORA-00942 报错怎么处理": {"p50": 6.1, "max": 6.3},
   "dense:limit=10": {"p50": 8.3, "max": 10.8},
   "dense:limit=50": {"p50": 12.8, "max": 15.6}
 },
 "builtin_hybrid_rrf_p50_ms": 20.8
}
```

## 附录 C：与 v1 的对照（一句话版）

| 问题 | v1 答案 | v2 答案 |
| --- | --- | --- |
| BM25 用谁 | SQLite FTS5 | Milvus 内建 BM25 + jieba |
| 分词 | 自写单双字 | jieba（黑盒，无法内省） |
| 能回退到无向量依赖吗 | 能（`provider=bm25`） | **不能**（保留旧镜像） |
| 索引体积 | 67.6 MB | 218–239 MB |
| 依赖新增 | 0 | 359 MB |
| 稀疏一路延迟 | 0.3 ms | 6–14 ms |
| 调优常数 | 全部生效 | 导航/版本保留，全词/同义词删除，四列权重近似 |
| 结论来源 | 文档 + wheel 元数据推测 | **真机实测** |