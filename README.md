# OceanBase DB Agent

A conversational agent for troubleshooting OceanBase database issues and optimizing SQL performance (**FastAPI Backend** + **Vue 3 Frontend**).

Users ask questions in natural language (e.g., "What are the slow SQLs?"), and the backend uses langchain + langgraph to orchestrate the LLM and tools. It retrieves OCP (OceanBase Control Platform) metadata and read-only SQL in real time, returning diagnostic results via **SSE streaming**. The frontend provides a chat interface built with Vue 3 + Vite.

The backend leverages langchain's `create_agent` under the hood to orchestrate tools, using a **dual-adapter architecture** for the OCP client and SQL executor.

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Directory Structure](#directory-structure)
- [Local Development (Mock mode recommended, offline)](#local-development-mock-mode-recommended-offline)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Configuration](#configuration)
  - [config.yaml Fields](#configyaml-fields)
  - [Configuration Priority](#configuration-priority)
- [Conversation Memory (PostgreSQL)](#conversation-memory-postgresql)
- [Tool Trace, Audit and Access Control](#tool-trace-audit-and-access-control)
- [Deployment (Production)](#deployment-production)
  - [Overall Topology](#overall-topology)
  - [Step 0: Prepare Runtime Data (Important)](#step-0-prepare-runtime-data-important)
  - [Step 1: Backend Deployment](#step-1-backend-deployment)
  - [Step 2: Frontend Build and Static Asset Hosting](#step-2-frontend-build-and-static-asset-hosting)
  - [Step 3: Reverse Proxy and /api Forwarding](#step-3-reverse-proxy-and-api-forwarding)
- [Production Example: Nginx + systemd](#production-example-nginx--systemd)
- [Integration Confirmation Checklist](#integration-confirmation-checklist)

---

## Tech Stack

**Backend**
- Python ≥ 3.13
- FastAPI (0.141) · Uvicorn · Pydantic v2
- langchain 1.4 / langchain-core 1.6 / langgraph 1.2 / langchain-openai (OpenAI-compatible LLM interface)
- httpx (OCP real client) · PyMySQL (SQL real executor) · PyYAML · python-dotenv
- Dependencies and locked versions can be found in `backend/requirements.txt`

**Frontend**
- Vite · Vue 3.5 · markdown-it · highlight.js · DOMPurify · vitest (Testing)

**Other**
- OceanBase Distributed Database Enterprise Edition 4.2.5 · OCP Enterprise Edition 4.2.5

---

## Directory Structure

```
ob_agent/
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── main.py           # create_app assembly (Config/LLM/Tools/Routes)
│   │   ├── config.py         # Config loading (YAML + .env + Env variables)
│   │   ├── sse.py            # SSE frame serialization
│   │   ├── agent/            # Agent orchestration
│   │   │   ├── runner.py     # create_agent event stream → User event stream (Context compression/Timeout/Confirmation)
│   │   │   ├── model.py      # Build ChatOpenAI
│   │   │   ├── prompt.py     # System Prompt (DBA Assistant + Rules)
│   │   │   ├── confirm.py    # HITL Human-in-the-loop confirmation channel (ConfirmationBroker + Middleware)
│   │   │   ├── tools.py      # 11 DBA tools + search_docs/read_doc + 2 doc file tools (15 registered)
│   │   │   ├── doc_index.py  # ob_wiki FTS5 index: build / search / read-by-section
│   │   │   ├── plan_view.py  # Normalize an OCP plan payload into a pre-order operator view
│   │   │   ├── plan_diff.py  # Diff two plan views: verdict / cost ratio / regressed operators
│   │   │   └── tool_input.py # Tool input Pydantic models
│   │   ├── api/
│   │   │   ├── chat.py       # POST /api/chat (SSE) · GET /api/health
│   │   │   ├── confirm.py    # POST /api/chat/confirm (Approval back-channel)
│   │   │   ├── threads.py    # GET /api/threads · GET/DELETE /api/threads/{id}
│   │   │   ├── audit.py      # GET /api/audit (tool-call audit)
│   │   │   └── deps.py       # Shared deps: Bearer token, memory availability, thread_id
│   │   ├── memory/           # PG checkpointer + chat_message history + audit_event
│   │   └── tools/            # Dual-adapter implementation
│   │       ├── base.py       # Data models / Exceptions / Protocol interfaces
│   │       ├── ocp/          # OCP client: mock.py (fixtures) / real.py (httpx)
│   │       └── sql/          # SQL executors: guard.py (read-only) / base (shared pool) / mock · mysql · oracle
│   ├── run.sh                # Startup entrypoint (creates .venv from requirements.txt on first run)
│   ├── requirements.txt
│   ├── config.example.yaml   # Example config (Tracked in Git)
│   ├── config.yaml           # Actual config (Gitignored, not tracked)
│   ├── .env.example          # Example env vars (Tracked in Git)
│   └── .env                  # Actual env vars (Gitignored, not tracked)
│   ├── scripts/              # unpack_doc.py: unzip the doc corpus in place (fixes filename encoding)
│   ├── eval/                 # Retrieval eval set + run_eval.py (CI quality gate)
│   ├── doc/                  # ob_wiki.zip (tracked) + ob_wiki/ (docs unzipped in place, gitignored)
│   └── data/                 # mock fixtures (Tracked; real_*.json is gitignored)
├── frontend/                 # Vue 3 + Vite frontend
│   ├── src/                  # Components / composables / api / lib
│   ├── tests/                # vitest pure logic unit tests
│   ├── index.html · vite.config.js · package.json
│   └── dist/                 # npm run build output
├── tests/                    # Backend pytest (Repository root, pytest.ini testpaths=tests)
├── .github/workflows/ci.yml  # CI: backend pytest + retrieval eval gate + frontend vitest
└── README.md
```

> Note: Production and local artifacts such as `backend/doc/` (except `ob_wiki.zip`), `backend/config.yaml`, `backend/.env`, `frontend/dist/`, `.venv/`, `node_modules/`, etc., are excluded by `.gitignore` and **will not** be distributed via git. See [Deployment Steps](#step-0-prepare-runtime-data-important).

---

## Local Development (Mock mode recommended, offline)

### Backend

Execute in the **repository root directory**:

```bash
# One-time installation: Create backend/.venv and install runtime + dev dependencies (pytest, etc.)
pip install -r requirements.txt
```

Start the service (defaults to mock mode if `backend/config.yaml` / `.env` is absent; can start even without LLM configured):

```bash
# Option 1: Start with uvicorn
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Option 2: use the repo script (works from any cwd; creates backend/.venv and
# installs from requirements.txt on first run, then serves on 127.0.0.1:8000)
./backend/run.sh
```

Health check:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expected output (Mock default, LLM unconfigured):

```json
{"status":"ok","ocp_provider":"mock","sql_provider":"mock","llm_configured":false,"memory_enabled":false,"auth_enabled":false}
```

Chat example (SSE streaming; returns a clear 503 error if LLM is not configured):

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"有哪些慢SQL？"}]}'
```

> With conversation memory enabled (`memory.enabled: true`) this same request must also carry a `thread_id`; see [Conversation Memory (PostgreSQL)](#conversation-memory-postgresql).

> When LLM is not configured, the service starts as usual, and `/api/chat` returns a 503 advising you to complete the LLM configuration. Offline end-to-end validation is completed by injecting a stub model (`tests/helpers/scripted_model.py`), without relying on a real LLM.

**Mock fixtures** — `backend/data/*.json` holds the offline fixtures that make `provider: mock` work end to end:

| Fixture | Used by |
| --- | --- |
| `ocp_tenants.json`, `ocp_clusters.json` | `get_tenant_info`, `get_cluster_list` |
| `ocp_cluster_stats.json`, `ocp_server_stats.json` | `get_cluster_resource_stats`, `get_server_resource_stats` |
| `ocp_slow_sqls.json` (first `sqlId` is `sq-scan-orders-1`) | `get_slow_sql` |
| `ocp_sql_text.json`, `ocp_top_plan.json`, `ocp_sql_explain.json` | `get_full_sql_text`, `get_sql_top_plan`, `get_sql_explain` |
| `ocp_sql_explain_after.json` + the second entry of `ocp_top_plan.json` | `compare_plans` (the same SQL after its index was lost: a regression fixture) |
| `sample_tables.json` | `execute_sql`, plus the synthesized `SHOW CREATE TABLE` behind `get_table_ddl` |
| `slow_sqls.json` | the mock `oceanbase.gv$sql_audit` path behind `execute_sql` |
| `explain_results.json` | the mock `EXPLAIN` lookup |

With `ocp.provider: mock` **and** `sql_ro.provider: mock`, all eleven DBA tools answer from these fixtures, so the demo runs with neither OCP nor a database reachable. `real_*.json` files are for captured real responses and stay gitignored.

### Frontend

```bash
cd frontend
npm install        # First time
npm run dev        # http://127.0.0.1:5173 (/api is proxied → 127.0.0.1:8000)
```

Start the backend first as described above (defaults to mock mode), then start the frontend. Entering "有哪些慢SQL？" on the webpage exhibits two behaviors: when LLM is unconfigured, it follows the **503 error branch** (displaying "LLM not configured" at the top); after connecting a real or stub LLM, entering the same prompt demonstrates the complete demo loop: `status` grey text → markdown streaming → `done`.

### Tests and CI

```bash
# Backend tests — run from the repository ROOT (pytest.ini sets testpaths=tests, pythonpath=backend tests)
backend/.venv/bin/python -m pytest -q

# Document retrieval eval gate — real corpus, real numbers (see backend/eval/README.md)
python backend/scripts/unpack_doc.py        # unpack the corpus once (idempotent; needed by the eval)
python backend/eval/run_eval.py --strict    # exit 0 pass / 2 below threshold / 3 stale eval set

# Frontend tests
cd frontend && npm test
```

Document tool tests fall into two layers: `tests/test_doc_index.py` drives a small synthetic corpus (fast, mechanism-level), while `tests/test_retrieval_eval.py` runs 50 real questions against the unpacked 5146-document corpus and gates on recall@5 and MRR — it skips itself when the corpus is absent. Baseline on the real corpus: **recall@1 60%, recall@5 82%, recall@10 94%, MRR@10 0.704**, ~114 ms per query; the gate fails below recall@5 80% / MRR 0.68, so a ranking tweak cannot silently degrade retrieval. `.github/workflows/ci.yml` runs exactly these steps (backend pytest → eval gate → frontend vitest) on pushes to `main` / `feature-dev` and on pull requests, uploading the full eval report as an artifact.

---

## Configuration

Backend configuration sources: **Environment Variables > `backend/config.yaml` > Default Values**; empty string environment variables will not override non-empty values in YAML. Support for `backend/.env` is also included (via dotenv). Only example files are committed to the repository; actual files should be generated locally.

```bash
cp backend/config.example.yaml backend/config.yaml
cp backend/.env.example backend/.env
```

### config.yaml Fields

| Section   | Field                                                    | Description                                      |
| --------- | -------------------------------------------------------- | ------------------------------------------------ |
| `ocp`     | `provider`                                               | `mock \| real` (**Note:** Set to `mock` for local demo) |
| `ocp`     | `base_url`                                               | OCP 4.3.5 gateway address (e.g., `https://<ocp-host>:<port>`) |
| `ocp`     | `username`/`password`                                    | HTTP Basic Auth (OCP admin account, not /login session) |
| `ocp`     | `verify_ssl`                                             | Whether to verify OCP TLS certificates           |
| `sql_ro`  | `provider`                                               | `mock \| real`                                  |
| `sql_ro`  | `host_map`                                               | When real: Cluster name → Connection string mapping (dict string) |
| `sql_ro`  | `username`/`password`                                    | Read-only database account (SELECT privilege only recommended) |
| `sql_ro`  | `connect_timeout` / `query_timeout_seconds` / `max_rows` | Connection/query timeout and max row limits      |
| `sql_ro`  | `driver`                                                 | **Oracle-mode tenants only** (ignored for MySQL): OCI driver (`oracledb` \| `cx_oracle`). Neither the DSN service name nor the tenant suffix is configurable — both come from the tool call: `db_name` is used as the DSN service name, and the read-only account is completed to `user@tenant#cluster` when it carries no `@`. `connect_timeout` / `query_timeout_seconds` are passed only to drivers that support them (`oracledb` thin mode: `tcp_connect_timeout` / `call_timeout`; `cx_Oracle`: no `tcp_connect_timeout` (skipped), `call_timeout` applied when the version allows it, otherwise skipped with a debug log) |
| `llm`     | `base_url`/`api_key`/`model`                             | OpenAI-compatible model API (All three required to be considered "configured") |
| `llm`     | `temperature` / `max_input_tokens`                       | Sampling temperature / Context compression threshold benchmark |
| `agent`   | `send_row_data`                                          | Whether results passed to LLM contain row data  |
| `agent`   | `max_seconds`                                            | Fallback limit for total execution time of a single agent turn |
| `agent`   | `confirm_db_ops`                                         | Whether to enable human approval for `execute_sql` (HITL) |
| `agent`   | `confirm_timeout_seconds`                                | Approval timeout (Defaults to rejection on timeout). **Must be less than `max_seconds`** — startup fails fast otherwise, since an equal/larger value would never fire before the whole-turn timeout |
| `agent`   | `recursion_limit`                                        | Maximum recursion steps for langgraph           |
| `memory`  | `enabled`                                                | Persist conversations to PostgreSQL (`true` when PG is available) |
| `memory`  | `host`/`port`/`user`/`password`/`dbname`                 | PostgreSQL connection; set `dsn` to override the individual parts |
| `memory`  | `pool_min_size` / `pool_max_size`                        | Connection pool bounds (shared by checkpointer and history table) |
| `memory`  | `list_limit` / `messages_limit`                          | Default page caps for the thread-list / history endpoints |
| `memory`  | `audit_retention_days`                                   | `>0` prunes audit rows older than N days at startup; `0` (default) keeps the trail forever |
| `memory`  | `open_timeout_seconds` / `open_attempts`                 | Startup connect budget; worst-case startup block ≈ `open_attempts × open_timeout_seconds` + backoff |
| `auth`    | `enabled`                                                | Require `Authorization: Bearer <token>` on every `/api/*` route except `/api/health` |
| `auth`    | `token`                                                  | The shared token; `enabled: true` with an empty token fails fast at startup |

Environment variables with the same names use uppercase format (e.g., `OCP_PROVIDER`, `LLM_BASE_URL`, `SEND_ROW_DATA`, `MEMORY_ENABLED`, `MEMORY_DSN`, `MEMORY_HOST`, `MEMORY_PASSWORD`, `AUTH_ENABLED`, `AUTH_TOKEN`, `CONFIRM_DB_OPS`, `CONFIRM_TIMEOUT_SECONDS`, `RECURSION_LIMIT`).

### Configuration Priority

```
Environment Variables (non-empty) > backend/config.yaml > Code Defaults
```

When LLM is not configured: backend starts normally, `/api/chat` returns `503`; `llm_configured` in `/api/health` shows `false`.

---

## Conversation Memory (PostgreSQL)

When `memory.enabled` is `true`, conversations are persisted so the agent remembers earlier turns and the frontend can browse history. Two stores share **one** PostgreSQL connection pool:

| Store | Role |
| --- | --- |
| LangGraph checkpoint (`AsyncPostgresSaver`, keyed by `thread_id`) | The agent state the LLM sees — this **is** the short-term memory. `setup()` creates `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` on first start. |
| `chat_message` (created by the backend) | The user/assistant transcript that powers the history UI. It is separate because `SummarizationMiddleware` rewrites checkpoint state once the context window fills up, which would silently drop early turns from the checkpoint. |

**Request contract change**: when memory is enabled, `POST /api/chat` requires `thread_id` and only the **last** message is treated as the new turn (earlier turns come from the checkpoint). The frontend generates a `thread_id` per conversation and remembers it in `localStorage`. With `memory.enabled: false` the old stateless contract applies: the client sends the full history and no `thread_id` is needed.

```bash
# First turn on a new conversation (later turns reuse the same thread_id)
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'content-type: application/json' \
  -d '{"thread_id":"demo-1","messages":[{"role":"user","content":"有哪些慢SQL？"}]}'

# History browsing
curl -s http://127.0.0.1:8000/api/threads
curl -s http://127.0.0.1:8000/api/threads/demo-1/messages
curl -X DELETE http://127.0.0.1:8000/api/threads/demo-1
```

`GET /api/threads` returns `{thread_id, title, created_at, updated_at, message_count}` (title = first user message, truncated). `DELETE` removes both the `chat_message` rows and the thread's checkpoints.

**Prerequisites**: a reachable PostgreSQL database and a role allowed to create tables in `public` (the checkpointer creates its own tables on first start). `AsyncPostgresSaver.setup()` is idempotent; with multiple uvicorn workers a concurrent first start may log a migration conflict, which is handled by treating already-created tables as ready.

**Degraded mode**: if PostgreSQL is unreachable, or `memory.enabled: false`, the backend still starts normally. `/api/threads*` returns `503`, `GET /api/health` reports `memory_enabled: false`, and the frontend hides the history sidebar and falls back to sending the full history on every request.

---

## Tool Trace, Audit and Access Control

### Tool trace

Every tool call is surfaced over SSE as a `tool` event once it finishes:

```json
{"type":"tool","phase":"end","id":"01a0ca10","name":"execute_sql","label":"执行只读 SQL",
 "args":{"sql":"select ..."},"ok":true,"error":null,"rows":200,"truncated":true,
 "approved":true,"duration_ms":842}
```

String arguments (SQL text) are truncated to 500 characters, and **result rows are never included** — the event carries call metadata only. The UI renders it as a collapsible list under the answer, so a diagnosis can be **checked** instead of trusted. Existing `status` events are unchanged, so older clients keep working.

`approved` is `true`/`false` only for tools gated by human approval (`execute_sql`), and `null` for everything else. A denied or timed-out approval never reaches the tool handler and therefore never produces an `on_tool_end` — the confirmation middleware emits the `tool` event itself, which is why a **rejected** operation still shows up in the trace.

`get_sql_explain` events additionally carry a `plan` object: the backend normalizes the OCP plan payload into a pre-order, depth-annotated operator tree plus an operator summary (operators, rows, cost, properties — still **no result rows**). The UI draws that as an execution-plan tree and highlights the most expensive operator; every other tool omits the field.

### Audit log

When memory is enabled, every tool event is also persisted to `audit_event` (same pool as the checkpointer): thread, tool, arguments, ok/error, rows, truncated, approved, duration.

```bash
curl -s "http://127.0.0.1:8000/api/audit?limit=20"
curl -s "http://127.0.0.1:8000/api/audit?thread_id=demo-1&tool=execute_sql"
```

The UI exposes the same data through the **工具审计** button (current conversation, or all conversations). The audit table deliberately stores SQL text — that is its purpose — but never result rows. Audit uses the memory connection pool, so if memory is unavailable `/api/audit` returns `503` while tool trace over SSE keeps working.

The audit trail is append-only: `DELETE /api/threads/{thread_id}` removes the conversation's messages and checkpoints but **intentionally keeps its audit rows**, so deleting a conversation cannot erase the record that a query was run.

### Access control

Minimal single-token scheme: set `auth.enabled: true` plus `auth.token` (or `AUTH_ENABLED`/`AUTH_TOKEN`). Every `/api/*` route except `/api/health` then requires `Authorization: Bearer <token>`; `/api/health` stays open so probes keep working. The frontend keeps the token in `localStorage` and prompts for it once on a `401` — no login page and no user accounts. Starting the backend with `auth.enabled: true` and an empty token fails immediately (fail closed).

> Because conversations are now persisted, an unprotected deployment lets anyone who can reach the port read and delete every conversation **and** trigger read-only SQL on your clusters. Enable the token unless the port is already restricted to trusted networks.

### Failure surfacing

An unexpected agent failure is **not** streamed verbatim: the client gets a generic message plus a short `error_id` (`{"type":"error","error_id":"ab12cd34","message":"agent 执行出错…（error_id=ab12cd34）"}`) while the full traceback stays in the server log — connection strings and hosts must not leak to the browser. Tool/database errors are the deliberate exception: they still flow to the LLM and into the trace and audit, because a DBA needs to see *why* a query failed.

Two concurrent requests on the same `thread_id` are rejected with `409` rather than being allowed to write divergent checkpoints, so a second browser tab gets a clear "this conversation is busy" message instead of silently corrupting the context.

`GET /api/health` reports `auth_enabled` always, and adds `memory_error` **only** when memory/audit degraded, so you can tell why the history endpoints are returning `503`.

---

## Deployment (Production)

> Dockerfile / docker-compose are **not included** in the repository. Production deployment utilizes the **Uvicorn (Backend) + Static Hosting + Reverse Proxy** architecture. The frontend build output consists of purely static files, and the API uses relative path `/api/*`. Therefore, a Web server is required to host static assets and forward `/api` to the backend, forming a unified site.

### Overall Topology

```
                         443/80
  Browser  ──────────────►  Nginx (Hosts frontend/dist static assets)
                                │   /api/*  →  127.0.0.1:8000
                                ▼
                            Uvicorn (app.main:app, Backend)
                                │
                       ┌────────┴────────┐
                       ▼                 ▼
                  OCP Gateway     Read-only Database
                 (ocp.base_url)       (sql_ro)
```

### Step 0: Prepare Runtime Data (Important)

The following data **is not distributed via git** (excluded by `.gitignore`) and must be manually placed under the `backend/` working directory during deployment:

1. **`backend/config.yaml`** and **`backend/.env`**: Generate and populate with actual values according to [Configuration](#configuration).
2. **`backend/doc/`**: OceanBase official documentation knowledge base. The repository ships the compressed archive `backend/doc/ob_wiki.zip`; unzip it in place before starting the backend, which yields `backend/doc/ob_wiki/` (the extracted files are gitignored). The agent reaches it through the `search_docs` / `read_doc` tools (full-text search + section-level reading), backed by the SQLite FTS5 index `backend/doc/ob_wiki.index.db` — a **build artifact** that is gitignored and rebuilt automatically (~5 s for 5100+ docs) whenever it is missing or the corpus changes. `read_file` / `list_directory` are still available for browsing and are rooted at `./doc`. **Missing this directory will cause document retrieval features to fail**.
3. **PostgreSQL** (only when `memory.enabled: true`): a reachable instance plus a role allowed to create tables in `public`. The backend creates its own checkpoint and history tables on first start. See [Conversation Memory (PostgreSQL)](#conversation-memory-postgresql).

> Working directory convention: The backend runs with `backend/` as its working directory (`run.sh` will `cd` to the script directory), ensuring relative paths like `./doc` and `./config.yaml` work properly.

### Step 1: Backend Deployment

**A. Install dependencies and set up venv**

```bash
cd backend
python3.13 -m venv .venv            # Or use uv
source .venv/bin/activate
pip install -r requirements.txt      # requirements.txt is the single source of dependencies
```

In production, it is recommended to **disable reload** and run continuously.

**B. Start with Uvicorn**

```bash
cd /path/to/ob_agent/backend
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

- `--host 127.0.0.1`: The backend listens locally only; external reverse proxy (Nginx) exposes ports 443/80 externally.
- `--workers`: **Keep this at 1.** The HITL confirmation channel (the frontend posts a decision to `/api/chat/confirm` with only a `request_id`) and the per-thread serialization lock are both **in-process** state. With more than one worker, an approval can land on a worker that never held the pending request (404/503), and two workers can run the same thread concurrently against the shared Postgres checkpointer. Horizontal scaling therefore requires sticky routing by `thread_id` at the gateway plus single-writer guarantees per thread — not provided by this version.
- Maintain `cd backend` throughout to ensure relative paths `./doc` and `./config.yaml` remain valid.

### Step 2: Frontend Build and Static Asset Hosting

```bash
cd frontend
npm ci            # Precise installation according to package-lock.json
npm run build     # Output generated in frontend/dist/
```

Copy the entire `frontend/dist/` directory to the deployment machine (or point Nginx root/alias to this directory). The build output consists of purely static files and does not depend on a Node runtime.

### Step 3: Reverse Proxy and /api Forwarding

The frontend initiates requests using relative paths:

- `POST /api/chat` (SSE streaming conversation)
- `POST /api/chat/confirm` (Approval back-channel)
- `GET /api/health` (Health check; the only route exempt from the access token)
- `GET /api/threads` · `GET /api/threads/{thread_id}/messages` · `DELETE /api/threads/{thread_id}` (Conversation history; 503 when memory is disabled)
- `GET /api/audit` (Tool-call audit log; 503 when memory is disabled)

The reverse proxy MUST:
1. Return static assets (HTML/JS/CSS, etc.) from `frontend/dist`.
2. Forward `/api/*` to backend `127.0.0.1:8000`.
3. **Disable buffering for `/api`** (SSE requires real-time forwarding; avoid Nginx buffering breaking stream immediacy).

---

## Production Example: Nginx + systemd

### 1) Nginx Site Configuration

`/etc/nginx/conf.d/ob-agent.conf`:

```nginx
server {
    listen 80;
    server_name your-domain.example.com;   # Replace with real domain/IP

    root /var/www/ob-agent;                # Directory where frontend/dist was copied
    index index.html;

    # Static assets (Includes history route fallback, index.html suffices for single-page app)
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API reverse proxy (Critical: SSE requires proxy buffering disabled)
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        # SSE: Disable proxy buffering to ensure real-time streaming
        proxy_buffering off;
        proxy_cache off;
        # Relax streaming/long-connection timeout
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

> For production, it is recommended to add TLS (Let's Encrypt / Certificates) on port 443 and use `proxy_pass http://127.0.0.1:8000;` inside `location /api/` (Note that the URL has no trailing `/` to preserve the original `/api` URI prefix).

### 2) systemd Backend Service

`/etc/systemd/system/ob-agent-backend.service`:

```ini
[Unit]
Description=OceanBase DB Agent backend (uvicorn)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/ob_agent/backend        # Must be backend; relative paths rely on it
ExecStart=/opt/ob_agent/backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=3
User=www-data                                 # Adjust based on actual execution user
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ob-agent-backend
sudo systemctl status ob-agent-backend
```

### 3) Post-Deployment Verification

```bash
# Direct backend health check
curl -s http://127.0.0.1:8000/api/health

# Full path health check via Nginx
curl -s https://your-domain.example.com/api/health

# SSE streaming conversation (via Nginx)
curl -N -X POST https://your-domain.example.com/api/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"有哪些慢SQL？"}]}'
```

Open the site in a browser, and you should see the empty state page "Hello, I am the OceanBase DBA Assistant". The health badge in the top-right corner displays `ocp_provider / sql_provider · LLM ready/unconfigured` (corresponding to fields returned by `GET /api/health`).

---

## Integration Confirmation Checklist

- **real OCP**: Populate endpoints and authentication according to OCP 4.3.5 official docs (currently a NotImplementedError skeleton).
- **real SQL**: EXPLAIN plan semantics, large result set cursor (SSCursor), `ob_query_timeout`, and read-only account permission scope.
- **SSE**: Verify server truly aborts task when client disconnects (no orphan tasks).
- **LLM**: After configuration, inject mock/demo hints per provider (the prompt no longer embeds mock hints).
- **Resource water level**: Implemented — added `get_cluster_list`, `get_cluster_resource_stats` (`GET /api/v2/ob/clusters/{id}/stats`, a flat `ClusterResourceStats` object) and `get_server_resource_stats` (`GET /api/v2/ob/clusters/{id}/serverStats`, a `data.contents` list). The tool layer whitelists the fields and derives the `cpuAssignedPct` / `memoryAssignedPct` / `dataDiskUsedPct` / `logDiskUsedPct` water levels; when nothing is returned it reports `ok:false` with `error_kind: not_found`, so "no data" is never read as "zero usage". Remaining live-test item: real payload field names match the docs (CPU in cores, memory/disk in bytes) and whether a sampling time window must be passed.
- **Document retrieval**: Implemented — `search_docs` / `read_doc` replace "guess directory names, then read whole files" (`backend/app/agent/doc_index.py`). The corpus is pre-tokenized for Chinese (CJK unigram + bigram) into an FTS5 external-content index (`tokenize='unicode61'`), so 2-character queries (`事务`, `索引`, `锁`) match; `trigram` cannot. Search returns the matching *section* with a readable snippet and a `score`; the MySQL/Oracle mode and the version written in the question are auto-detected and used as filters (the same-named MySQL/Oracle documents are otherwise indistinguishable); Chinese question words / fillers (`哪些` / `如何` / `一共` / `包含` …) are dropped from the query terms, so "错误码一共有哪些" no longer surfaces FAQ pages that merely say 哪些 a lot; navigation files (`index.md` and the root `README.md`) are classified as `nav` and heavily down-weighted (`NAVIGATION_FILE_PENALTY`) instead of excluded — they only point at documents, and the body text is authoritative — while `include_index=true` lifts the penalty for genuine "what categories are there / how is the doc set organised" questions; in-document navigation sections (`相关文档` / `参见` / `更多信息`) are down-weighted too. `read_doc` then returns just the requested section plus a `sections` TOC. Measured on the real corpus: 5146 docs → 25077 chunks, 67.7 MB index, ~5 s rebuild, ~70–130 ms per query (the OR-expanded round dominates).
- **Plan regression detection**: Implemented — `compare_plans` (`backend/app/agent/plan_diff.py`) answers "the same SQL suddenly got slower, why?". It rebuilds both plan trees, converts OCP's *cumulative* cost into per-operator self cost (so the true culprit is the leaf that changed, not every ancestor that inherits the increase), aligns siblings by `(operator, name)` with an LCS, and reports a verdict (`unchanged` / `changed` / `regressed` / `improved`), the cost ratio, the regressed and improved operators, added/removed operators, and property-level notes (available index lost, `physical_range_rows` jump, index-back, output rows). Unrelated branches are pruned from the returned tree. On the mock regression fixture (index lost → full scan) it reports `regressed`, cost ratio 95.2, cost 1958 → 186416 and rows 1 → 971070, pointing at `PHY_TABLE_SCAN(WRT(WARN_RULE_TOTAL_INDEX_N1))`.
- **Retrieval eval + CI**: Implemented — `backend/eval/` holds 50 real DBA questions with expected documents (`retrieval_cases.jsonl`) and `run_eval.py`, which reports recall@1/@5/@10, MRR, latency and per-tag breakdown, and exits non-zero when recall@5 < 80% or MRR < 0.68 (baseline: 60% / 82% / 94%, MRR 0.704). `tests/test_retrieval_eval.py` runs the same gate inside pytest plus two invariants: every expected path must still exist in the corpus (a stale eval set fails loudly after a corpus upgrade), and navigation pages must never take the top slot from a body document. `.github/workflows/ci.yml` runs backend pytest, this gate, and frontend vitest; `backend/scripts/unpack_doc.py` unpacks the corpus in CI (it repairs the archive's non-UTF-8 filenames and pins mtimes to the archive so the index fingerprint is machine-independent).
- **Oracle Tenant**: Implemented — `execute_sql` / `get_table_ddl` now route Oracle-mode tenants to the OCI driver (`backend/app/tools/sql/oracle.py`). The DSN service name comes from the `db_name` argument of the tool (the tenant's SERVICE_NAME), not from configuration; the read-only account is completed to `user@tenant#cluster` when it carries no `@`. `get_table_ddl` passes `db_name` as the DDL owner and falls back to `all_tab_columns where owner = <db_name>`. `connect_timeout` / `query_timeout_seconds` are only passed to drivers that support them: `oracledb` thin mode takes both, `cx_Oracle` has no `tcp_connect_timeout` (skipped) and `call_timeout` is applied only when the version allows it, with the skip logged at debug level. Remaining live-test item: whether the tenant's SERVICE_NAME equals the `db_name` you pass (a mismatch surfaces as ORA-12514/12505), and whether `DBMS_METADATA.GET_DDL` is available. Note `cx_Oracle` has no wheel for Python ≥ 3.11, so the default `driver` is `oracledb` (thin mode, no Oracle client libraries needed).