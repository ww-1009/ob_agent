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
| Milvus sparse | ≥0.78 | ≥0.70 | 对照基线 79.33% / 0.706，不退化即可；**M6 实测 88.27% / 0.768（通过）** |
| Milvus hybrid | ≥0.85 | ≥0.72 | 起步值，M6 实测后再收紧（等 embedding 配额） |

外加：`literal` 档退化 ≤1 条（≥19/28）、导航页抢 top1 越界 = 0、故障不得 500（embedding 挂
只跑稀疏；Milvus 打不开返回空结果 + `retrieval_degraded`；rerank 超时保留融合序）。
hybrid 的延迟门禁等 Linux 实测后再定（API embedding 单次约 +210ms，旧「hybrid P50 ≤150ms」不可达）。

### M6 实测：Milvus sparse（2026-09，Linux，同一 179 条）

```
--retriever sparse --max-nav-top1 0
命中率@1  70.39%   命中率@5  88.27%   命中率@10  92.18%   MRR@10  0.768
延迟 ms: 均值 86.1 / P50 67.5 / P95 126.3        导航页抢 top1: 0
by_tag@5: body 91.67% / hard 63.64% / list 45.45% / literal 92.86%(26/28) / mode 100% / nav 100% / version 100%
```

相对 FTS5 基线：命中率@5 79.33% → 88.27%、MRR 0.706 → 0.768、@10 87.71% → 92.18%，
`literal` 19/28 → **26/28**，`list` 54.55% → 45.45%（唯一退化的档，见下）。
延迟 P50 与 FTS5（63.6ms）接近，构建到 25077 块只要 73s（`--no-vectors`）。

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

## 指纹 TTL 缓存（延迟基线口径）

`search` 每次都要遍历全库 stat 一遍算语料指纹（5146 篇实测 **52.4ms**，曾占端到端近一半）。
现在指纹带 TTL 缓存（`retrieval.fingerprint_ttl_seconds`，默认 5s；wiki 根目录 mtime 变化、
`ensure(force=True)` 都会立刻失效）。同一台机器上同一套用例：

| | 均值 | P50 | P95 |
| --- | --- | --- | --- |
| 缓存前 | 116.6ms | 118.0ms | 151.0ms |
| 缓存后 | 62.1ms | 61.3ms | 101.3ms |

代价是往**已有子目录**里新增/修改文件时，索引最迟 TTL 秒后才重建（新增子目录会改根目录
mtime，能立刻发现）；需要立刻生效就用 `ensure(force=True)` 或把 TTL 设为 0。

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