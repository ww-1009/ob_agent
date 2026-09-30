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
| Milvus sparse | ≥0.78 | ≥0.70 | 对照基线 79.33% / 0.706，不退化即可；**M6 实测 88.27% / 0.768、M7 重录 88.27% / 0.760（通过）** |
| Milvus dense | ≥0.78 | ≥0.70 | 对照 sparse 不退化即可；**M7 首次实测 86.59% / 0.763（通过）** |
| Milvus hybrid | ≥0.85 | ≥0.72 | 起步值，按 M7 实测可收紧；**M7 首次实测 91.62% / 0.819（通过）** |

外加：`literal` 档退化 ≤1 条（≥19/28）、导航页抢 top1 越界 = 0、故障不得 500（embedding 挂
只跑稀疏；Milvus 打不开返回空结果 + `retrieval_degraded`；rerank 超时保留融合序）。
延迟门禁等 M8 加 `--max-p50-ms` 一类开关后再定：Linux 实测（179 条、空载、新进程 reopen）
sparse P50 89.1ms / dense 175.0ms / hybrid 275.5ms —— hybrid 每次查询都要调 embedding API，
延迟与配额都是它比 sparse 多出来的成本。

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
  退回纯守卫（构建侧 `truncated` 只在配置值被压小时才出现）。
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

## 指纹 TTL 缓存（已随 FTS5 一起删除）

FTS5 时代 `search` 每次都要遍历全库 stat 一遍算语料指纹（5146 篇实测 **52.4ms**，曾占端到端
近一半），于是加了 `retrieval.fingerprint_ttl_seconds` TTL 缓存（缓存前 P50 118.0ms → 缓存后
61.3ms，见 git 历史）。**M7 删掉 FTS5 后这条路径整体不存在了**：索引新鲜度由
`milvus_index --incremental` 的语料指纹短路负责（构建期一次），检索请求不再扫语料目录。

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