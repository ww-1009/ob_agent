# 文档检索评测（P4）

`search_docs` 的排序是一堆手调常数（bm25 权重、全词命中奖励、同义词惩罚、版本奖励、导航惩罚），
改任何一个都可能让某一类问题悄悄变差——几个 e2e 用例看不出来。这个目录把「检索好不好用」
变成可回归的数字，并在 CI 里做门禁。

## 跑一次

```bash
# 语料是解压产物、不入库，先解压（已解压会跳过）
python backend/scripts/unpack_doc.py

# 全量评测 + 门禁（退出码 0 通过 / 2 低于阈值 / 3 语料或用例有问题）
python backend/eval/run_eval.py

# 只看明细，附完整报告 json 供 diff
python backend/eval/run_eval.py --json /tmp/retrieval-eval.json --show 20

# 改排序公式时：对两条分支各跑一次，比 json
python backend/eval/run_eval.py --json /tmp/before.json

# 引擎 / 重排对照（M8）：--retriever sparse|dense|hybrid，--rerank auto|off|api
# 「无重排」基线必须显式 --rerank off。另外 auto 自 M9 起只在 sparse 上生效：非 sparse 传 auto
# 会被检索层护栏降成 off 并在报告里记 rerank_skipped_cases，比不出差距是护栏在起作用，不是配置没读上
python backend/eval/run_eval.py --retriever sparse --rerank off --json /tmp/m8_sparse_off.json
python backend/eval/run_eval.py --retriever hybrid --rerank api --json /tmp/m8_hybrid_api.json

# 门禁开关：--min-hit1（命中率@1，重排与默认引擎的验收线）、--max-p50-ms（延迟，只在固定机器上开）
python backend/eval/run_eval.py --retriever hybrid --rerank api --strict --min-hit1 0.65
```

pytest 里的门禁（`tests/test_retrieval_eval.py`）就是同一套：语料不存在时整体跳过，
存在时校验用例、跑指标、并额外断言**导航页不得顶掉正文答案**（①②③ 的核心保证）。

## 指标口径

| 指标 | 含义 | 为什么关心 |
| --- | --- | --- |
| 命中率@1 | 正确答案直接排第一的比例 | 最接近模型与用户的体验 |
| 命中率@5 | 前 5 条里有正确答案的比例（k 与工具默认 `limit=5` 对齐） | 模型一次能看到 5 条 |
| 命中率@10 | 前 10 条里的比例 | 放宽后还有多少是「排序问题」而非「召不回」 |
| MRR@10 | 首个命中排名倒数的均值（未命中记 0） | 顺序敏感：答案排第 8 名和第 1 名差很多 |

用例里的 `expect` 是**路径子串**而不是精确文件名：同一个问题往往有几篇都算对
（MySQL / Oracle 双份文档、总览与分册），判分要认这些。冒烟时用 `--limit N` 只跑前几条。

## 基线（2026-09，5146 篇真语料，179 条用例，FTS5 现状）

```
命中率@1  63.69%   命中率@5  79.33%   命中率@10  87.71%   MRR@10  0.706
延迟 ms: 均值 66.9 / P50 63.6 / P95 110.7
```

分档命中率@5（`by_tag`，`tag` 见下）：

| tag | 条数 | 命中率@5 | 用途 |
| --- | --- | --- | --- |
| `version` | 7 | 100.00% | 提问自带版本号（4.2.5 / V3.x→V4.x） |
| `mode` | 12 | 91.67% | MySQL / Oracle 同名文档消歧 |
| `body` | 60 | 85.00% | 正文细节型自然提问 |
| `nav` | 5 | 80.00% | 术语 / FAQ / 简介 / 快速上手 |
| `literal` | 28 | 67.86% | 错误码、视图名、配置项/系统变量、函数名 |
| `list` | 11 | 54.55% | 清单/总览类（`include_index=true`） |
| `hard` | 11 | 54.55% | 间接措辞、需要跨文档推断 |
| （无） | 45 | — | 首版遗留用例，未分档 |

门禁阈值取「基线往下留一点」：`--min-recall 0.76`（命中率@5）、`--min-mrr 0.68`。
它们只用来拦**退化**，不代表满意——命中率@5 79.33% 说明还有明显的提升空间。

`literal` 是后续引擎迁移的主对照档：Milvus 稀疏一路（内建 BM25）迁移后，要求这一档
**相对本条基线退化不超过 1 条**（即 ≥19/28）。延迟数字带指纹 TTL 缓存（见下）。

### 引擎迁移门禁（2026-09 决定，对照本页 179 条基线）

Milvus 迁移不沿用旧文档按 50 条用例写的 80%/88%，一律对照本页基线：

| 通道 | 命中率@5 | MRR@10 | 说明 |
| --- | --- | --- | --- |
| 现状 FTS5 | ≥0.76 | ≥0.68 | 现有 `--min-recall/--min-mrr` 默认值；实测 79.33% / 0.706 |
| Milvus sparse | ≥0.80 | ≥0.68 | 对照基线 79.33% / 0.706，不退化即可；**M6/M7 实测 88.27% / 0.760（通过）**；M8 起 CI 通道 A 就按这一行门禁 |
| Milvus dense | ≥0.78 | ≥0.70 | 对照 sparse 不退化即可；**M7 首次实测 86.59% / 0.763（通过）** |
| Milvus hybrid | ≥0.88 | ≥0.78 | 起步 0.85/0.72，M8 按 M7 实测收紧；**M7 实测 91.62% / 0.819**，M8 四通道复测一致（通过） |
| Milvus hybrid + rerank api | ≥0.88 | ≥0.75，另加命中率@1 ≥0.65 | 夜检通道 C 的探针；门槛比 B 松（0.784 距 0.78 只剩 0.004，会被重排接口抖动抖红）；**实测 90.50% / 0.784 / @1 70.39%（通过，但见下节的负面结论）** |

外加：`literal` 档退化 ≤1 条（≥19/28）、导航页抢 top1 越界 = 0、故障不得 500（embedding 挂
只跑稀疏；Milvus 打不开返回空结果 + `retrieval_degraded`；rerank 超时保留融合序；语料变了但索引没重建 → `index_stale`，结果照给）。
延迟门禁 M8 已加开关（`--max-p50-ms`），但**只在固定 nightly 机器上开**：延迟与机器强相关，
本机读数（179 条、空载、新进程 reopen）是 sparse P50 89.1ms / dense 175.0ms / hybrid 275.5ms /
hybrid+rerank 236.2ms(P95 421.3) —— hybrid 每次查询都要调 embedding API，延迟与配额都是它比
sparse 多出来的成本，所以设计文档 §12.2 写的 P50 ≤150ms 一类数字对本机不成立，阈值要按实测留 headroom 定。

### M6 实测：Milvus sparse（2026-09，Linux，同一 179 条）

```
--retriever sparse --max-nav-top1 0
命中率@1  70.39%   命中率@5  88.27%   命中率@10  92.18%   MRR@10  0.768
延迟 ms: 均值 86.1 / P50 67.5 / P95 126.3        导航页抢 top1: 0
by_tag@5: body 91.67% / hard 63.64% / list 45.45% / literal 92.86%(26/28) / mode 100% / nav 100% / version 100%
```

相对 FTS5 基线：命中率@5 79.33% → 88.27%、MRR 0.706 → 0.768、@10 87.71% → 92.18%，
`literal` 19/28 → **26/28**，`list` 54.55% → 45.45%（唯一退化的档，见下）。
延迟 P50 与 FTS5（63.6ms）接近，构建到 25077 块只要 73s（`--no-vectors`；M7 重录 25220 块 / 75s）。

### M7 实测：删 FTS5 + 分块根治（2026-09，Linux，同一 179 条）

```
--retriever sparse --max-nav-top1 0（索引用 M7 新分块 --rebuild --no-vectors 重建）
命中率@1  68.72%   命中率@5  88.27%   命中率@10  92.74%   MRR@10  0.760
延迟 ms: 均值 87.4 / P50 67.5 / P95 125.2        导航页抢 top1: 0
by_tag@5: body 91.67% / hard 63.64% / list 45.45% / literal 92.86%(26/28) / mode 100% / nav 100% / version 100%
门禁: 命中率@5 88.27% >= 76.00% ✓   MRR 0.760 >= 0.680 ✓   导航页抢 top1 0 <= 0 ✓
```

分块根治后语料 25077 → **25220 块**（doc 23995 → 24138，+143 块全来自下面那批 H1-only
文件；构建 `truncated` 6 → 0），全量 `--rebuild --no-vectors` 75.4s —— **CI 门禁现在先跑
这一步**（`.github/workflows/ci.yml`，全新 checkout 没有索引库，不建库会整体降级成空结果）。
相对 M6：命中率@5 持平 88.27%、@10 92.18% → 92.74%（被截断的长文件正文回到索引里），
MRR 0.768 → 0.760（多出来的长尾块让少数用例名次后移，仍远高于 0.70 门禁）。

- **删 FTS5**：`doc_index.py` 925 → 461 行，检索只剩 Milvus 三路，`retrieval.default_retriever`
  默认 **`sparse`**（hybrid 已过门禁，但默认值换成它要另做产品取舍，见下）；`read_doc` 改为直接读 wiki 文件、不依赖任何
  索引库——能看到整篇文档的全部小节（FTS5 时代 `chunks` 表每篇最多 2 块，读出来是残缺的）。
- **分块根治**：`_split_chunks` 对**没有 H2/H3 的长文件**原来走「`h1 or 正文` 整篇一块」兜底，
  绕过了 1800 字切分，只能按 `max_text_bytes` 截断（实测 H1-only 且正文 >1800 字的共 55 篇，
  其中 6 篇 >8000 字，最大 `组件 & 工具/运维管理/obshell/错误码.md` 40199 字 → 23 块）。
  M7 起兜底路径同样做 1800 字硬切（空行优先），内容不再丢，`max_text_bytes`
  退回纯守卫（构建侧 `truncated` 只在配置值被压小时才出现）。该值同时写进 `ob_meta.text_max_length`，
  构建期与 `--verify` 都会校验：改了 `max_text_bytes` 即 schema 不一致（VARCHAR 上限只在建集合时取一次，
  对已有集合不生效）——不带向量则整集重建成新上限，带向量则要求 `--rebuild-vectors`；`--stats` 会打印这一键。
- 代价：这批文件的 pk 与文本变了，索引必须 `--rebuild` 一次（已做，数字见上），
  `literal` 档 26/28 不受影响。

### M7 实测：三路对照（同一索引、同一 179 条，dense/hybrid 首次测出）

向量索引建好后（`--rebuild`，24138 块真向量 + 1082 块导航零向量，`embedded=24138`、`swapped=true`、
`verify=ok`、962s），三路各跑一遍完整评测：

```
sparse  命中率@1 68.72%  @5 88.27%  @10 92.74%  MRR 0.760   mean 121.2 / P50 89.1 / P95 151.9 ms
dense   命中率@1 68.16%  @5 86.59%  @10 92.18%  MRR 0.763   mean 203.4 / P50 175.0 / P95 304.1 ms
hybrid  命中率@1 75.98%  @5 91.62%  @10 94.41%  MRR 0.819   mean 318.4 / P50 275.5 / P95 398.2 ms
by_tag@5  sparse                     / dense                     / hybrid
  body    91.67%                     /  88.33%                   /  96.67%
  hard    63.64%                     /  54.55%                   /  81.82%
  list    45.45%                     /  54.55%                   /  36.36%   ← 唯一 hybrid 更差的档
  literal 92.86% (26/28)             /  96.43% (27/28)           /  96.43% (27/28)
  mode    100% / nav 100% / version 100%  / 100% / 85.71% / 100%  / 100% / 100% / 100%
导航页抢 top1: 三路均 0    门禁: 三路全过（hybrid 91.62% / 0.819 达 M8 前的 ≥0.85/≥0.72）
```

- **hybrid 是质量最好的一路**：@5 +3.35pt、MRR +0.059、@1 +7.26pt（相比 sparse），`hard` 档
  63.64% → 81.82% 提升最明显（RRF 用向量补上稀疏的字面量/措辞差）；代价是 P50 89 → 276ms、
  每次查询都要调 embedding API（配额/可用性风险，见下面的降级策略）。
- **dense 单跑不如 sparse**（86.59% vs 88.27%），但 `literal` 96.43% 与 `list` 54.55% 更好；
  它的价值在融合里，不在单跑。
- **`list` 档是 hybrid 的短板**（45.45% → 36.36%）：清单类提问要的是「总览/索引页」，
  向量召回会把普通正文页顶上来，RRF 之后反而挤掉索引页；`include_index=true` 的用例
  目前靠 sparse 的路由偏好兜底，M8 可以考虑「清单类问法只走 sparse」。
- 产品默认仍是 **`sparse`**（`retrieval.default_retriever`）：hybrid 过门禁了，但每天每查一次的
  API 调用与 +187ms P50 是产品取舍，不由评测结果单方面决定。
- 注：sparse 在有向量的索引上 P50 89.1ms，比 `--no-vectors` 索引上的 67.5ms 慢约 20ms
  （同一进程要驻留 99MB 向量 + IVF 索引），仍远好于 FTS5 时代带 10x 伪影的读数。

### M8 实测：API 重排（P2a，四通道对照）

同一索引、同一 179 条用例（`--max-nav-top1 0`），`--rerank api` 走真实 DashScope
（`qwen3.7-text-rerank`、`top_n=15`、`passage = "标题 > 小节\n正文(≤500 字)"`）：

```
通道               命中率@1  @5      @10     MRR@10  延迟 mean/P50/P95(ms)   重排动过 top1
sparse  --rerank off  68.72%  88.27%  92.74%  0.760   121.2 / 89.1 / 151.9      0
sparse  --rerank api  69.27%  89.94%  92.74%  0.776   272.0 / 236.2 / 421.3    43
hybrid  --rerank off  75.98%  91.62%  94.41%  0.819   318.4 / 275.5 / 398.2      0
hybrid  --rerank api  70.39%  90.50%  94.41%  0.784   466.2 / 439.9 / 664.2    40
```

**结论：重排没有达到设计 §11.2 的「hit@1 相对新基线 ≥ +5pp」，而且在 hybrid 上是净损害。**

- **sparse**：@1 +0.55pp、@5 +1.67pp、MRR +0.016，**`literal` 26/28 → 27/28**、`mode`@1 83.33%→91.67%、
  `version`@1 71.43%→85.71%、`nav`@1 80%→100%；代价是 `body`@1 80%→75%、`list`@1 36.36%→27.27%。
  净微正，但要多花 **+147ms P50**（外部 API 往返）—— 值不值是产品取舍。
- **hybrid**：@1 **−5.59pp**、MRR −0.035，`literal`@1 85.71%→71.43%、`body`@1 83.33%→75%、`mode`@1 100%→91.67%；
  变好的是 `hard`@1 27.27%→54.55%、`list`（fixture 都在 top5 内）。**别在 hybrid 上开重排。**
- **调参探针（hybrid，k=deep=5）**：变 passage 与 `top_n` 都救不回来 —— 只留正文（去掉标题前缀）
  崩到 @1 **55.31%**；把 `max_passage_chars` 加到 1000 是 67.60%；`top_n=5` 是最不坏的一档
  （@1 71.51%、@5 91.62% 与不重排持平、MRR 0.797），但仍低于不重排的 75.98%。
  → 这不是 passage 构造的 bug，而是「重排模型的相关性口径」与评测口径（要的正是那一篇答案文档）
  不一致：融合序已经很强时，重排把靠语义相似上来的页面顶掉了。救法得是**融合分与重排分混合**
  （像列权重重排那样按 β 混），不是纯换序 —— 留作 P3，M8 不动。
- 结论落到配置上：`rerank.mode` 默认 `auto`（配置齐了就重排）**只在 sparse 上成立**。
  这一条已从「文档自律」升为**代码护栏**：`MilvusRetriever.search` 在 `retriever != sparse`
  且模式为 `auto` 时把重排降成 `off`，并把原因写进 `RetrievalResult.rerank_skipped`
  （报告里的 `rerank_skipped_cases`）；显式 `--rerank api` 不受影响，所以
  夜检通道 C 仍然是先跑生产口径的 `sparse + rerank api` 门禁，再跑 `hybrid + rerank api` 作探针。

做到这一步靠两件事（设计取舍见设计文档 §17）：

1. **客户端列权重重排**：Milvus 内建 BM25 只有一列 `text`（标题/关键词/小节/正文揉在一起），
   拿不到 FTS5 的列权重。构建时除加权文本外**另存 `keywords` 一列**（`SCHEMA_VERSION=2`），
   检索期按 FTS5 同口径 `(title 10, keywords 6, section 4, body 1)` 算覆盖率，
   以 `0.5 * 归一化距离 + 0.5 * 列覆盖率` 重排；**只对稀疏一路**打分（稠密一路只有原文语义，
   打列分会让向量召回失效），列分用**原始查询**（同义词扩展只帮召回、不参与列分）。
2. **保留同义词查询扩展**：缺列信号时扩展有害（hit@5 76.5%），补上列分后扩展有益（87.7% → 88.3%）。

`TEXT_MATCH` 过滤这条近路走不通：Milvus Lite 里它是 **OR** 语义（`TEXT_MATCH(text,"日志流 管理")`
比单词命中还多），`minimum_should_match` 直接语法报错，且带它的一次检索要 **4.8s**。

已知的弱项（M6 实测，属 P2/P3，尚未修）：

- `list` 档 45.45%（5/11）：`include_index=true` 的清单类提问要的是「总览/索引页」，
  但 45 个无档位用例里也有这类问法；期望文档常只是目录下的普通 doc，列分帮不上。
- 完全未命中 14 条：`config-overview`、`cluster-resource`、`merge-memory`、`lit-ora-04031`、
  `list-error-code-cat`、`list-pl-pkg`、`list-config-sysvar`、`list-perf-tuning`、
  `body-arbitration-intro`、`body-ha-overview`、`body-ls-manage`、`hard-merge-oom`、
  `hard-drop-table-recover`、`hard-ora-01555-longquery`
  —— FTS5 的「AND 全词路由」在 Milvus 里没有廉价等价物（见上），长尾精确匹配仍在往下掉。
- 「命中但排在 5 名之后」7 条（括号内是名次）：`error-code`(10)、`lit-mysql-6002`(9)、
  `grant`(8)、`list-mysql-views`(8)、`body-mem-tool`(8)、`body-tenant-capacity`(6)、
  `hard-conn-timeout`(6)。

**延迟必须按「构建 → close → 新进程 reopen 再测」的口径取**：milvus-lite v3 是纯 Python 进程内
实现，在建索引的那个进程里测会得到约 10x 的伪影（同一目录 26000 行：构建进程内 sparse P50 779ms /
dense 599ms，reopen 后 109ms / 27ms）。测量期间机器上也不能有别的重活——2 vCPU 上并发两个 Milvus
进程同样把 P50 从 ~80ms 推到 ~800ms。Linux 权威读数见 `docs/probes/probe7.out.json` 与设计文档 §5.3。

FTS5 基线期的弱项（保留作历史对照，其中 `lock-wait`、`backup-overview`、`lit-ob-query-timeout`
等在 Milvus sparse + 列权重重排后已进前 5）：

- 错码/范围页这类「一条文档覆盖几百个错误码」的页，字面量查询常被更短的页面挤掉：
  `lit-ora-04031`（至今仍未命中）、`lit-ob-query-timeout`、`lit-gv-sysstat`。
- `version-425`（4.2.5 新增特性）：答案《V4.2.5 文档更新记录》输给《版本发布记录》汇总页。
- `list` 档与长尾排序不稳，详见上面的 M6 实测清单。

**改了排序公式、语料或 `SCHEMA_VERSION` 之后**：先跑一次看数字，确认提升再下调阈值；
如果是有意取舍（某类变好、另一类变差），把两条曲线都写进 PR 说明再调阈值。

## 语料新鲜度（查询期）

构建期靠 `milvus_index --incremental` 的指纹短路决定「是否跳过」；但**查询期**一度完全不检测：
语料更新了、索引没重建，检索会静默返回旧结果（设计文档 §10 曾把它记为遗留缺口）。现在
`MilvusRetriever.search` 成功后调 `MilvusIndex.corpus_changed(ttl_seconds=...)`：读 `ob_meta.corpus_dir`
复算语料指纹，与建库时写的 `corpus_fingerprint` 比对（只比前三个字段 `文件数:总字节:最新mtime`——
线上指纹还带 `:t{标题重复}k{关键词重复}` 的加权口径签名，那一段无法从语料复算），不等则置
`RetrievalResult.index_stale = True` 并打一条 warning；`tools.py` 把它翻成「请重建索引」的 hint。
**结果照给、永不 500**，`index_stale` 与 `degraded` 是并列标记（两者能同时成立）。

- **代价**：真语料 5146 篇扫一遍约 **51ms**，所以按 `retrieval.fingerprint_ttl_seconds`（默认 5.0，
  0 = 每次重扫）缓存；这条检查是旁路，内部任何异常只记 debug 日志，不影响检索结果。
- **不误报**：老库没有 `corpus_dir`、目录被移走、扫不出文件（`0:0:0`）都当「没证据」→ 不报过期。
- 测试：`tests/test_milvus_index.py` 的 `corpus_changed` 五例（新鲜 / 改语料 / 无目录 / 目录不存在 /
  TTL 缓存）、`tests/test_retrieval.py::test_real_index_flags_stale_corpus`（真库端到端）、
  `tests/test_doc_index.py::test_search_docs_tool_reports_index_stale`（hint 组装）。

## 加用例

往 `retrieval_cases.jsonl` 里加一行 JSON：

```json
{"id": "kill-session", "query": "怎么终止一个会话", "expect": ["终止租户会话"], "mode": "", "version": "", "include_index": false, "tag": ""}
```

- `id` 唯一、短横线命名；`query` 写成用户真会问的样子（口语、带错误码、带版本号都可以）。
- `expect` 至少一条，且**必须能在语料里匹配到路径**——否则 `--strict` 会直接失败，
  避免评测集随着语料升级悄悄过期（新加用例先跑 `--strict` 验证）。
- `mode` / `version` 留空即自动识别；`include_index: true` 用于「有哪些 / 包含哪些」清单类提问。
- `tag` 可选，取值约定：`literal`（精确字面量）、`mode`（模式消歧）、`version`（带版本号）、
  `list`（清单类）、`nav`（术语/FAQ/简介）、`body`（正文细节）、`hard`（间接措辞）。
  报告按 tag 分组统计（`by_tag`），引擎迁移时用它逐档对比退化。

## CI 三通道（M8）

`.github/workflows/ci.yml` 在 M8 拆成三条评测通道 —— 它们的**成本**差别很大，全塞进每个 PR 会拖垮 CI：

| 通道 | 何时跑 | 检索 | 密钥 | 门禁 |
| --- | --- | --- | --- | --- |
| A `backend` | 每 push / PR | `--retriever sparse --rerank off` | 不需要 | `--min-recall 0.80 --min-mrr 0.68 --max-nav-top1 0` |
| B `hybrid` | 每 push / PR | `--retriever hybrid --rerank off` | 不需要（走向量缓存） | `--min-recall 0.88 --min-mrr 0.78 --max-nav-top1 0` |
| C `retrieval-nightly` | `schedule`（20:00 UTC）/ 手动 | ① `--retriever sparse --rerank api`（生产口径）② `--retriever hybrid --rerank api`（探针） | embedding + rerank | ① `--min-recall 0.88 --min-mrr 0.75 --min-hit1 0.66` ② `--min-recall 0.88 --min-mrr 0.75 --min-hit1 0.65` |

- **A 恒跑**：稀疏索引纯本地构建（25220 块 ~75s，无密钥），顺序是「建库 → pytest → 评测」——
  pytest 里的三条门禁断言（`tests/test_retrieval_eval.py`）与 CLI 门禁共用同一份索引。
- **B 靠 `actions/cache`**：向量索引 24138 条向量、本机全量构建 ~16 分钟、约 9M token，不可能每 PR 现建。
  缓存键 `milvus-vectors-${{ hashFiles('backend/doc/ob_wiki.zip') }}-v1`（换语料或换嵌入模型时**记得一起改后缀**）。
  **缓存未命中时 B 明确跳过并打 `::notice::`，不假绿**；夜检跑过一次后各分支都能命中（默认分支的缓存对全仓可见）。
- **C 只在夜检 / 手动**：用 `secrets.EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` 与
  `secrets.RERANK_BASE_URL` / `RERANK_API_KEY`（fork 的 PR 拿不到密钥，所以不能放进每 PR 通道）。
  这五个缺任何一个都会跳过 + `::warning::`（`BASE_URL` / `MODEL` 也查，是因为 workflow env 写成空串会
  覆盖配置里的值，会在建库/重排时报一个容易被误读的错）。向量建好后**单独一步** `actions/cache/save@v4`：
  这样即使后面的重排门禁失败，下一个 PR 的 B 也已有向量可用（用 `actions/cache@v4` 的话 job 失败即不保存）。
  `RERANK_PROTOCOL: dashscope` 必须显式给 —— 配置默认是 `jina`，不给会拼出错端点。

**要在仓库里配的 Actions secrets**：`EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL`、
`RERANK_BASE_URL` / `RERANK_API_KEY` / `RERANK_MODEL`（可选，默认 `qwen3.7-text-rerank`）。
阈值一律取「实测基线往下留一档」，只为拦**退化**；语料/模型/公式一变就重录基线再调。

## 运维速查（M8）

```bash
cd backend
# 1) 建索引：只建稀疏（最快）/ 含向量（dense、hybrid 及 API 重排评测都需要）
.venv/bin/python -m app.agent.milvus_index --rebuild --no-vectors   # ~75s，无需密钥
.venv/bin/python -m app.agent.milvus_index --rebuild                # ~16min，约 9M token
.venv/bin/python -m app.agent.milvus_index --incremental            # 日常增量（语料指纹未变则短路）
.venv/bin/python -m app.agent.milvus_index --health --json          # rows / dims / has_vectors / 降级标志
.venv/bin/python -m app.agent.milvus_index --verify                 # 自检（--rebuild 末尾已内置）

# 2) 起服务：--workers 必须是 1（HITL 确认通道、同 thread 串行锁、Milvus Lite 连接都是进程内状态）
bash run.sh
```

- **启动预热**：`backend/app/main.py` 的 lifespan 会异步跑一次稀疏检索（`_warmup_retrieval`；索引库不存在时静默跳过），
  把 jieba 分词器加载与 Milvus Lite 首次查询的固定开销（实测首查 2027.8ms，其中分词器 571ms）挪到启动期。
  只预热稀疏一路，不引入启动期的外部 API 依赖。
- **降级怎么排**：`search_docs` 的返回带 `degraded` 与 `hint`（`backend/app/agent/tools.py` 的 `_DEGRADED_HINTS`）。
  `milvus_index_missing` → 按上面第 1 步建库；`milvus_unavailable` → 确认 `--workers 1`、没有别的建索引进程占着库；
  `index_stale` → 语料比索引新（`ob_meta` 里记的指纹与现在复算的对不上）→ `--rebuild`；TTL 内（默认 5s）不重扫，刚改完语料可能晚一拍才报；`dense_unavailable` → 稠密一路不可用（embedding 挂了），只走了稀疏，建议换核心词提问；
  `rerank_degraded=True` 但 `degraded=""` → 结果按融合序返回，质量略降但功能可用。
  `rerank_skipped="retriever_hybrid|retriever_dense"` → 不是失败，是护栏按「auto 只在 sparse
  上成立」主动没排（要在这两路上量重排得显式 `--rerank api`）。
- **回滚演练**：v2 没有 `RETRIEVAL_PROVIDER=fts5` 这种一键开关（FTS5 代码已删），退路是**镜像**：
  `git checkout 27ea070`（tag `fts5-final`）配同一份 `backend/doc/` 起服务，它读 `backend/doc/ob_wiki.index.db`。
  上线后**至少保留一个发布周期的旧镜像与该索引文件**（67.6MB）。另外还有两档配置级止损，不用回滚代码：
  `RETRIEVAL_DEFAULT_RETRIEVER=sparse`（免掉每次查询的 embedding 调用）与 `RERANK_MODE=off`（免掉外部重排调用）。
- **回滚演练（M8 实测，2026-09-30）**：在 `git worktree add --detach /tmp/m8_rollback 27ea070` 的干净检出里
  软链同一份 `backend/doc/ob_wiki`，三步全部通过：

  | 步骤 | 命令 | 实测 |
  | --- | --- | --- |
  | 1 | `python -m app.agent.doc_index --rebuild --stats` | FTS5：**5146 篇 / 25077 块 / 67.6MB，6s** |
  | 2 | `python eval/run_eval.py --strict`（旧口径） | 命中率@5 **79.33% ≥ 76%**、MRR **0.706 ≥ 0.68** → **PASS**，25.3s |
  | 3 | `python -m pytest -q`（旧测试套件） | **344 passed**，154.2s |

  结论：tag `fts5-final` 能在不依赖 Milvus 的情况下独立重建索引并过旧门禁，回滚路径可用。
  注意旧检出里的 `backend/doc/ob_wiki.index.db` 与 v2 的 `ob_wiki.milvus.db` **同名目录下互不干扰**，演练不会污染现网索引。