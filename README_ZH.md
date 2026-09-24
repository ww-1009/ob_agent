# OceanBase DB Agent

排查 OceanBase 数据库问题、优化 SQL 性能的对话式 agent（**FastAPI 后端** + **Vue 3 前端**）。

用户用自然语言提问（如“有哪些慢SQL？”），后端基于 langchain + langgraph 编排 LLM 与工具，实时检索
OCP（OceanBase 管控平台）元数据与只读 SQL，并以 **SSE 流式**返回诊断结果；前端为 Vue 3 + Vite 的聊天界面。

后端底层用 langchain `create_agent` 编排工具，OCP 客户端与 SQL 执行器为**双适配器**

---

## 目录

- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [本地开发（推荐 mock 模式，离线）](#本地开发推荐-mock-模式离线)
  - [后端](#后端)
  - [前端](#前端)
- [配置说明](#配置说明)
  - [config.yaml 字段](#configyaml-字段)
  - [配置优先级](#配置优先级)
- [会话记忆（PostgreSQL）](#会话记忆postgresql)
- [工具轨迹、审计与访问控制](#工具轨迹审计与访问控制)
- [部署（生产环境）](#部署生产环境)
  - [整体拓扑](#整体拓扑)
  - [第 0 步：准备运行时数据（重要）](#第-0-步准备运行时数据重要)
  - [第 1 步：后端部署](#第-1-步后端部署)
  - [第 2 步：前端构建与静态资源托管](#第-2-步前端构建与静态资源托管)
  - [第 3 步：反向代理与 /api 转发](#第-3-步反向代理与-api-转发)
- [生产运行示例：Nginx + systemd](#生产运行示例nginx--systemd)
- [联调待确认清单](#联调待确认清单)

---

## 技术栈

**后端**
- Python ≥ 3.13
- FastAPI（0.141）· Uvicorn · Pydantic v2
- langchain 1.4 / langchain-core 1.6 / langgraph 1.2 / langchain-openai（OpenAI 兼容 LLM 接口）
- httpx（OCP real 客户端）· PyMySQL（SQL real 执行器）· PyYAML · python-dotenv
- 依赖与版本锁定见 `backend/requirements.txt`

**前端**
- Vite · Vue 3.5 · markdown-it · highlight.js · DOMPurify · vitest（测试）

**其他**
- Oceanbase分布式数据库企业版4.2.5 · OCP企业版4.2.5 

---

## 目录结构

```
ob_agent/
├── backend/                  # FastAPI 后端
│   ├── app/
│   │   ├── main.py           # create_app 装配（配置/LLM/工具/路由）
│   │   ├── config.py         # 配置加载（YAML + .env + 环境变量）
│   │   ├── sse.py            # SSE 帧序列化
│   │   ├── agent/            # Agent 编排
│   │   │   ├── runner.py     # create_agent 事件流 → 用户事件流（上下文压缩/超时/确认）
│   │   │   ├── model.py      # 构建 ChatOpenAI
│   │   │   ├── prompt.py     # System Prompt（DBA 助手 + 规则）
│   │   │   ├── confirm.py    # HITL 人工确认通道（ConfirmationBroker + 中间件）
│   │   │   ├── tools.py      # 7 个 DBA 工具 + 2 个只读文档文件工具（共注册 9 个）
│   │   │   └── tool_input.py # 工具入参 Pydantic 模型
│   │   ├── api/
│   │   │   ├── chat.py       # POST /api/chat（SSE）· GET /api/health
│   │   │   ├── confirm.py    # POST /api/chat/confirm（审批反向通道）
│   │   │   ├── threads.py    # GET /api/threads · GET/DELETE /api/threads/{id}
│   │   │   ├── audit.py      # GET /api/audit（工具调用审计）
│   │   │   └── deps.py       # 共享依赖：Bearer 令牌 / 记忆可用性 / thread_id 校验
│   │   ├── memory/           # PG 检查点 + chat_message 历史表 + audit_event 审计表
│   │   └── tools/            # 双适配器实现
│   │       ├── base.py       # 数据模型 / 异常 / 协议接口
│   │       ├── ocp/          # OCP 客户端：mock.py（fixtures）/ real.py（httpx）
│   │       └── sql/          # SQL 执行器：guard.py（只读防线）/ base（公共骨架）/ mock · mysql · oracle
│   ├── run.sh                # 启动入口（首次运行会据 requirements.txt 创建 .venv）
│   ├── requirements.txt
│   ├── config.example.yaml   # 示例配置（入库）
│   ├── config.yaml           # 实际配置（已 gitignore，不再被 git 跟踪）
│   ├── .env.example          # 示例环境变量（入库）
│   └── .env                  # 实际环境变量（已 gitignore，不再被 git 跟踪）
│   ├── ob_wiki/              # OceanBase 官方文档知识库（gitignore，运行时需就位）
│   └── data/                 # mock fixtures（已入库；real_*.json 被 gitignore）
├── frontend/                 # Vue 3 + Vite 前端
│   ├── src/                  # 组件 / composables / api / lib
│   ├── tests/                # vitest 纯逻辑单测
│   ├── index.html · vite.config.js · package.json
│   └── dist/                 # npm run build 产物
├── tests/                    # 后端 pytest（仓库根，pytest.ini testpaths=tests）
└── README.md
```

> 说明：`backend/ob_wiki/`、`backend/config.yaml`、`backend/.env`、`frontend/dist/`、`.venv/`、
> `node_modules/` 等生产/本地产物均已被 `.gitignore` 排除，**不会**通过 git 分发，见[部署注意事项](#第-0-步准备运行时数据重要)。

---

## 本地开发（推荐 mock 模式，离线）


### 后端

在**仓库根目录**执行：

```bash
# 一次性安装：创建 backend/.venv 并装入运行依赖 + dev 依赖（pytest 等，用于跑测试）
pip install -r requirements.txt
```

启动服务（默认读不到 `backend/config.yaml` / `.env` 时按 mock 默认运行，无 LLM 也可启动）：

```bash
# 方式一：uvicorn 启动
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 方式二：用仓库内脚本（任意目录均可执行；首次运行会自动创建 backend/.venv
# 并按 requirements.txt 安装，随后监听 127.0.0.1:8000）
./backend/run.sh
```

探活：

```bash
curl -s http://127.0.0.1:8000/api/health
```

预期输出（mock 默认、未配置 LLM）：

```json
{"status":"ok","ocp_provider":"mock","sql_provider":"mock","llm_configured":false,"memory_enabled":false,"auth_enabled":false}
```

聊天示例（SSE 流式；未配置 LLM 时返回 503 清晰提示）：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"有哪些慢SQL？"}]}'
```

> 启用会话记忆（`memory.enabled: true`）时，同一请求还需带上 `thread_id`，见[会话记忆（PostgreSQL）](#会话记忆postgresql)。

> 未配置 LLM 时服务照常启动，`/api/chat` 返回 503 提示填写 LLM 配置；离线端到端验证由注入
> stub 模型（`tests/helpers/scripted_model.py`）完成，不依赖真实 LLM。

**mock 夹具** —— `backend/data/*.json` 是让 `provider: mock` 真正可用的离线夹具：

| 夹具 | 被谁使用 |
| --- | --- |
| `ocp_tenants.json`、`ocp_clusters.json` | `get_tenant_info` |
| `ocp_slow_sqls.json`（首条 `sqlId` 为 `sq-scan-orders-1`） | `get_slow_sql` |
| `ocp_sql_text.json`、`ocp_top_plan.json`、`ocp_sql_explain.json` | `get_full_sql_text`、`get_sql_top_plan`、`get_sql_explain` |
| `sample_tables.json` | `execute_sql`，以及 `get_table_ddl` 背后合成的 `SHOW CREATE TABLE` |
| `slow_sqls.json` | `execute_sql` 背后 mock 的 `oceanbase.gv$sql_audit` 路径 |
| `explain_results.json` | mock 的 `EXPLAIN` 匹配表 |

当 `ocp.provider: mock` **且** `sql_ro.provider: mock` 时，7 个工具全部改由这些夹具作答，因此在既连不上 OCP、也连不上数据库的环境下演示仍可跑通。`real_*.json` 用于存放抓取的真实响应，仍被 gitignore。

### 前端

```bash
cd frontend
npm install        # 首次
npm run dev        # http://127.0.0.1:5173（/api 已 proxy → 127.0.0.1:8000）
```

先按上文起好后端（mock 默认），再开前端。页面输入“有哪些慢SQL？”的行为分两种：未配置 LLM 时走
**503 错误分支**（顶部提示“LLM 未配置”）；接入真实或桩 LLM 后，同一输入才可见
`status` 灰字 → markdown 流式 → `done` 的演示闭环。

---

## 配置说明

后端配置来源：**环境变量 > `backend/config.yaml` > 默认值**；空字符串环境变量不覆盖 YAML 已有非空值。
也支持 `backend/.env`（dotenv 加载）。仓库只提交示例文件，实际文件由本地生成。

```bash
cp backend/config.example.yaml backend/config.yaml
cp backend/.env.example backend/.env
```

### config.yaml 字段

| 区块        | 字段                                                       | 说明                                              |
| --------- | -------------------------------------------------------- | ----------------------------------------------- |
| `ocp`     | `provider`                                               | `mock \| real`（**注意：** 本地 mock 演示请设为 `mock`）    |
| `ocp`     | `base_url`                                               | 填 OCP 4.3.5 网关地址（如 `https://<ocp-host>:<port>`） |
| `ocp`     | `username`/`password`                                    | HTTP Basic Auth（OCP 管理账号，非 /login 会话）           |
| `ocp`     | `verify_ssl`                                             | 是否校验 OCP TLS 证书                                 |
| `sql_ro`  | `provider`                                               | `mock \| real`                                  |
| `sql_ro`  | `host_map`                                               | real 时：集群名 → 连接串的映射（dict 字符串）                   |
| `sql_ro`  | `username`/`password`                                    | 只读数据库账号（建议仅授 SELECT）                            |
| `sql_ro`  | `connect_timeout` / `query_timeout_seconds` / `max_rows` | 连接/查询超时与结果行数上限                                  |
| `sql_ro`  | `driver`                                                 | **仅 Oracle 模式租户**（MySQL 忽略）：OCI 驱动（`oracledb` \| `cx_oracle`）。DSN 的 service name 与用户名里的租户后缀都不可配置，均来自工具调用：`db_name` 作 DSN 的 service name，只读账号无 `@` 时补成 `user@tenant#cluster`。`connect_timeout` / `query_timeout_seconds` 只传给支持它们的驱动（`oracledb` 瘦模式：`tcp_connect_timeout` / `call_timeout`；`cx_Oracle`：没有 `tcp_connect_timeout`（跳过），`call_timeout` 在版本支持时设置，否则跳过并记 debug 日志） |
| `llm`     | `base_url`/`api_key`/`model`                             | OpenAI 兼容模型接口（三者齐全才算“已配置”）                      |
| `llm`     | `temperature` / `max_input_tokens`                       | 采样温度 / 上下文压缩阈值基准                                |
| `agent`   | `send_row_data`                                          | 送入 LLM 的结果是否含行数据                                |
| `agent`   | `max_seconds`                                            | 单轮 agent 执行总时长兜底                                |
| `agent`   | `confirm_db_ops`                                         | 是否开启 `execute_sql` 人工审批（HITL）                   |
| `agent`   | `confirm_timeout_seconds`                                | 审批超时（超时默认拒绝）。**必须小于 `max_seconds`**：否则启动即失败——相等或更大时整轮超时总是先触发，审批超时永远轮不到 |
| `agent`   | `recursion_limit`                                        | langgraph 图最大递归步数                               |
| `memory`  | `enabled`                                                | 是否把会话持久化到 PostgreSQL（PG 可用时置 `true`）              |
| `memory`  | `host`/`port`/`user`/`password`/`dbname`                 | PG 连接信息；给 `dsn` 可整体覆盖分项                          |
| `memory`  | `pool_min_size` / `pool_max_size`                        | 连接池上下限（检查点与历史表共用一个池）                             |
| `memory`  | `list_limit` / `messages_limit`                          | 会话列表 / 历史消息接口的默认条数上限                            |
| `memory`  | `audit_retention_days`                                   | `>0` 时启动清理更早的审计行；`0`（默认）永久保留审计留痕                |
| `memory`  | `open_timeout_seconds` / `open_attempts`                 | 启动连接预算；最坏启动阻塞 ≈ `open_attempts × open_timeout_seconds` + 退避 |
| `auth`    | `enabled`                                                | 除 `/api/health` 外所有 `/api/*` 要求 `Authorization: Bearer <token>` |
| `auth`    | `token`                                                  | 共享令牌；`enabled: true` 而令牌为空会导致启动失败（fail closed）    |

环境变量同名键为大写形式（如 `OCP_PROVIDER`、`LLM_BASE_URL`、`SEND_ROW_DATA`、`MEMORY_ENABLED`、`MEMORY_DSN`、`MEMORY_HOST`、`MEMORY_PASSWORD`、`AUTH_ENABLED`、`AUTH_TOKEN`）。

### 配置优先级

```
环境变量（非空） > backend/config.yaml > 代码默认值
```

未配置 LLM 时：后端照常启动，`/api/chat` 返回 `503`；`/api/health` 的 `llm_configured` 为 `false`。

---

## 会话记忆（PostgreSQL）

`memory.enabled: true` 时会话会被持久化：Agent 能记住前几轮对话，前端也能浏览历史。同一套 PostgreSQL 连接池上放两个存储：

| 存储 | 作用 |
| --- | --- |
| LangGraph 检查点（`AsyncPostgresSaver`，按 `thread_id` 索引） | 模型看到的 Agent 状态，**就是**短期记忆本身。首次启动 `setup()` 会建 `checkpoints`、`checkpoint_blobs`、`checkpoint_writes`、`checkpoint_migrations`。 |
| `chat_message`（后端自建） | 前端历史界面用的 user/assistant 终稿记录。之所以单独存：上下文写满后 `SummarizationMiddleware` 会改写检查点里的消息，早期对话会从检查点中消失。 |

**请求契约变化**：启用记忆后 `POST /api/chat` 必须带 `thread_id`，且**只把 `messages` 的最后一条当本轮新消息**（更早的轮次由检查点提供）。前端为每个会话生成 `thread_id` 并记在 `localStorage`。`memory.enabled: false` 时沿用旧的无状态契约：前端回传全量历史，不需要 `thread_id`。

```bash
# 新会话第一轮（后续轮次复用同一个 thread_id）
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"thread_id":"demo-1","messages":[{"role":"user","content":"有哪些慢SQL？"}]}'

# 历史浏览
curl -s http://127.0.0.1:8000/api/threads
curl -s http://127.0.0.1:8000/api/threads/demo-1/messages
curl -X DELETE http://127.0.0.1:8000/api/threads/demo-1
```

`GET /api/threads` 返回 `{thread_id, title, created_at, updated_at, message_count}`（标题取首条用户提问并截断）。`DELETE` 会同时删除 `chat_message` 行与该 thread 的检查点。

**前置条件**：PG 可达，且账号能在 `public` 下建表（检查点首次启动自行建表）。`AsyncPostgresSaver.setup()` 幂等；多 worker 并发首次启动可能报迁移冲突，已按「表已存在即视为就绪」处理。

**降级行为**：PG 连不上或 `memory.enabled: false` 时，后端照常启动：`/api/threads*` 返回 `503`，`GET /api/health` 报 `memory_enabled: false`，前端隐藏历史侧栏并退回「每轮回传全量历史」的无状态模式。

---

## 工具轨迹、审计与访问控制

### 工具轨迹（tool trace）

每次工具调用结束时都会通过 SSE 发出一条 `tool` 事件：

```json
{"type":"tool","phase":"end","id":"01a0ca10","name":"execute_sql","label":"执行只读 SQL",
 "args":{"sql":"select ..."},"ok":true,"error":null,"rows":200,"truncated":true,
 "approved":true,"duration_ms":842}
```

字符串入参（SQL 文本）截断到 500 字符，且**绝不包含结果行数据**——只带调用元信息。前端把它渲染成答案下方的折叠列表，让诊断结论可以**被核对**而不是被信任。原有 `status` 事件保持不变，旧客户端不受影响。

`approved` 只对受人工审批管控的工具（`execute_sql`）取 `true`/`false`，其余为 `null`。被拒绝或审批超时的调用根本不会进入工具 handler，因此不会有 `on_tool_end`——由确认中间件自己补发 `tool` 事件，这也是**被拒绝的操作**同样会出现在轨迹里的原因。

`get_sql_explain` 的轨迹还会多一个 `plan` 字段：后端把 OCP 计划报文归一化成「先序、带 depth 的算子树 + 算子汇总」（算子、行数、代价、属性，同样**不含结果行数据**），前端据此在轨迹里画出执行计划树并标出代价占比最高的算子；其余工具不带该字段。

### 审计日志

启用记忆后，每条 tool 事件都会落库到 `audit_event`（与检查点同池）：thread、工具、入参、成败、行数、是否截断、是否被批准、耗时。

```bash
curl -s "http://127.0.0.1:8000/api/audit?limit=20"
curl -s "http://127.0.0.1:8000/api/audit?thread_id=demo-1&tool=execute_sql"
```

前端通过顶栏的**工具审计**按钮查看同样的数据（可切换「仅本会话 / 全部会话」）。审计表**有意保存 SQL 文本**（这正是它的用途），但从不保存结果行数据。审计复用记忆连接池，故记忆不可用时 `/api/audit` 返回 `503`，而 SSE 上的工具轨迹仍照常工作。

审计留痕只增不改：`DELETE /api/threads/{thread_id}` 会删掉会话的消息与检查点，但**有意保留其审计行**，因此删除会话无法抹掉「某次查询被执行过」这一记录。

### 访问控制

最小方案：单令牌。设置 `auth.enabled: true` 与 `auth.token`（或环境变量 `AUTH_ENABLED` / `AUTH_TOKEN`），此后除 `/api/health` 外所有 `/api/*` 都要求 `Authorization: Bearer <token>`；`/api/health` 保持放行以便探活。前端把令牌存在 `localStorage`，收到 `401` 时弹一次性输入框——不引入登录页，也不引入用户账号。`auth.enabled: true` 而令牌为空时后端**启动即失败**（fail closed）。

> 由于会话现在会被持久化，未加保护的部署意味着任何能访问该端口的人都能读取、删除全部会话，**并**对你的集群触发只读 SQL。除非端口已限定在可信网络内，请开启令牌。

### 失败如何呈现

agent 的**意外异常不会原文下发**：客户端只拿到一句通用文案加一个短 `error_id`（`{"type":"error","error_id":"ab12cd34","message":"agent 执行出错…（error_id=ab12cd34）"}`），完整堆栈只留在服务端日志——连接串、主机名不应泄漏给浏览器。工具/数据库错误是有意的例外：它们仍会送给 LLM 并落入轨迹与审计，因为 DBA 需要看到查询**为什么**失败。

同一 `thread_id` 的并发请求会返回 `409` 而不是被允许写出分叉的检查点，因此第二个浏览器标签会看到明确的「该会话正在处理中」，而不是静默写坏上下文。

`GET /api/health` 固定返回 `auth_enabled`，并**仅在**记忆/审计降级时附带 `memory_error`，便于判断历史接口为何返回 `503`。

---

## 部署（生产环境）

> 仓库内**未包含** Dockerfile / docker-compose，生产部署采用 **Uvicorn（后端）+ 静态托管 + 反向代理** 方案。
> 前端构建产物为纯静态文件，且 API 走相对路径 `/api/*`，因此需要一个 Web 服务器托管静态资源并把 `/api`
> 转发到后端，组成同一站点。

### 整体拓扑

```
                         443/80
  Browser  ──────────────►  Nginx（托管 frontend/dist 静态资源）
                                │   /api/*  →  127.0.0.1:8000
                                ▼
                            Uvicorn（app.main:app, 后端）
                                │
                       ┌────────┴────────┐
                       ▼                 ▼
                    OCP 网关         只读数据库
                 （ocp.base_url）    （sql_ro）
```

### 第 0 步：准备运行时数据（重要）

以下数据**不随 git 分发**（已被 `.gitignore` 忽略），部署时必须手动放置到
`backend/`` 工作目录下：

1. **`backend/config.yaml`** 与 **`backend/.env`**：按[配置说明](#配置说明)生成并填写真实值。
2. **`backend/ob_wiki/`**：OceanBase 官方文档知识库目录。System Prompt 约定文档入口为
   `./ob_wiki/README.md`，agent 通过文件工具（根目录限定在该目录）只读引用。**缺少该目录会导致文档检索功能不可用**。
3. **PostgreSQL**（仅当 `memory.enabled: true`）：可连的实例 + 能在 `public` 下建表的账号。检查点与历史表由后端首次启动时自建，见[会话记忆（PostgreSQL）](#会话记忆postgresql)。

> 运行目录约定：后端以 `backend/` 为工作目录运行（`run.sh` 会 `cd` 到脚本所在目录），
> 使 `./ob_wiki`、`./config.yaml`的相对路径生效。

### 第 1 步：后端部署

**A. 安装依赖并准备 venv**

```bash
cd backend
python3.13 -m venv .venv            # 或使用 uv
source .venv/bin/activate
pip install -r requirements.txt      # 依赖以 requirements.txt 为唯一来源
```

生产环境建议**关闭 reload** 并常驻运行。

**B. 用 Uvicorn 启动**

```bash
cd /path/to/ob_agent/backend
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

- `--host 127.0.0.1`：后端只监听本机，由外部反向代理（Nginx）对外提供 443/80。
- `--workers`：**必须保持 1。** 人工确认（HITL）通道（前端只带 `request_id` POST 到 `/api/chat/confirm`）与同一 thread 的串行锁都是**进程内**状态：多 worker 时审批可能落到从未持有该待确认请求的进程（404/503），且两个 worker 可能同时跑同一 thread 并写同一个 Postgres 检查点。需要横向扩展时，必须在网关按 `thread_id` 做粘性路由并自行保证同一 thread 单写——当前版本不提供该能力。
- 全程保持 `cd backend` 以确保 `./ob_wiki`、`./config.yaml` 相对路径正确。

### 第 2 步：前端构建与静态资源托管

```bash
cd frontend
npm ci            # 按 package-lock.json 精确安装
npm run build     # 产物输出到 frontend/dist/
```

将 `frontend/dist/` 整个目录拷贝到部署机（或作为 Nginx root/alias 指向该目录）。
产物为纯静态文件，不依赖 Node 运行时。

### 第 3 步：反向代理与 /api 转发

前端通过相对路径发起请求：

- `POST /api/chat`（SSE 流式对话）
- `POST /api/chat/confirm`（审批反向通道）
- `GET /api/health`（健康检查）
- `GET /api/threads` · `GET /api/threads/{thread_id}/messages` · `DELETE /api/threads/{thread_id}`（历史会话；记忆未启用时返回 503）
- `GET /api/audit`（工具调用审计；记忆未启用时返回 503）

反向代理必须：
1. 把静态资源（HTML/JS/CSS 等）从 `frontend/dist` 返回。
2. 把 `/api/*` 转发到后端 `127.0.0.1:8000`。
3. **关闭对 `/api` 的缓冲**（SSE 需要即时转发，避免 Nginx 缓冲导致流式不实时）。

---

## 生产运行示例：Nginx + systemd

### 1）Nginx 站点配置

`/etc/nginx/conf.d/ob-agent.conf`：

```nginx
server {
    listen 80;
    server_name your-domain.example.com;   # 替换为真实域名/IP

    root /var/www/ob-agent;                # frontend/dist 拷贝到的目录
    index index.html;

    # 静态资源（含 history 路由回退，本项目为单页无路由，index.html 即可）
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API 反向代理（关键：SSE 需关闭缓冲）
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        # SSE：禁用代理缓冲，保证流式即时
        proxy_buffering off;
        proxy_cache off;
        # 流式/长连接超时放宽
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

> 生产建议再叠加 TLS（Let’s Encrypt / 证书）到 443，并用 `location /api/` 的
> `proxy_pass http://127.0.0.1:8000;`（注意 URL 不含 `/`，保留原 URI 前缀 `/api`）。

### 2）systemd 后端服务

`/etc/systemd/system/ob-agent-backend.service`：

```ini
[Unit]
Description=OceanBase DB Agent backend (uvicorn)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ob_agent/backend        # 必须为 backend，相对路径依赖它
ExecStart=/opt/ob_agent/backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=3
User=www-data                                 # 按实际运行用户调整
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ob-agent-backend
sudo systemctl status ob-agent-backend
```

### 3）部署后验证

```bash
# 后端直连探活
curl -s http://127.0.0.1:8000/api/health

# 经 Nginx 走完整链路探活
curl -s https://your-domain.example.com/api/health

# SSE 流式对话（经 Nginx）
curl -N -X POST https://your-domain.example.com/api/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"有哪些慢SQL？"}]}'
```

浏览器打开站点，应看到“你好，我是 OceanBase DBA 助手”空态页，右上角健康徽章显示
`ocp_provider / sql_provider · LLM 就绪/未配置`（对应 `GET /api/health` 返回的字段）。

---



## 联调待确认清单

- **real OCP**：端点与鉴权按 OCP 4.3.5 官方文档填充（现为 NotImplementedError 骨架）。
- **real SQL**：EXPLAIN 计划语义、大结果集游标（SSCursor）、`ob_query_timeout` 与只读账号授权范围。
- **SSE**：客户端断开时确认服务端真中止（无孤儿 task）。
- **LLM**：配置完成后，mock/演示提示（system prompt 规则 6）改为按 provider 注入。
- **Oracle 租户**：已实现 —— `execute_sql` / `get_table_ddl` 现已把 Oracle 模式租户路由到 OCI 驱动（`backend/app/tools/sql/oracle.py`）。DSN 的 service name 直接取工具的 `db_name` 参数（即该租户的 SERVICE_NAME），不再从配置读取；只读账号无 `@` 时补成 `user@tenant#cluster`。`get_table_ddl` 以 `db_name` 作 DDL 的 owner，回退 SQL 为 `all_tab_columns where owner = <db_name>`。`connect_timeout` / `query_timeout_seconds` 只传给支持它们的驱动（`oracledb` 瘦模式两者都支持；`cx_Oracle` 无 `tcp_connect_timeout`（跳过），`call_timeout` 仅在版本支持时设置，跳过时记 debug 日志）。待联调确认：该租户的 SERVICE_NAME 是否与你传入的 `db_name` 一致（不一致会报 ORA-12514/12505）、以及是否开放 `DBMS_METADATA.GET_DDL`。注意 `cx_Oracle` 无 Python ≥ 3.11 轮子，故 `driver` 默认 `oracledb`（瘦模式，无需 Oracle 客户端库）。
