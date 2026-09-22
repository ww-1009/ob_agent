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
│   │   │   ├── tools.py      # 7 Tools exposed to LLM + ob_wiki file tool
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
│   │       └── sql/          # SQL executor: guard.py (Read-only defense) / mock / real
│   ├── run.py / run.sh       # Startup entrypoints
│   ├── requirements.txt
│   ├── config.example.yaml   # Example config (Tracked in Git)
│   ├── config.yaml           # Actual config (Gitignored, not tracked)
│   ├── .env.example          # Example env vars (Tracked in Git)
│   └── .env                  # Actual env vars (Gitignored, not tracked)
│   ├── ob_wiki/              # OceanBase official docs knowledge base (Gitignored, required at runtime)
│   └── data/                 # mock fixtures (Tracked; real_*.json is gitignored)
├── frontend/                 # Vue 3 + Vite frontend
│   ├── src/                  # Components / composables / api / lib
│   ├── tests/                # vitest pure logic unit tests
│   ├── index.html · vite.config.js · package.json
│   └── dist/                 # npm run build output
├── tests/                    # Backend pytest (Repository root, pytest.ini testpaths=tests)
└── README.md
```

> Note: Production and local artifacts such as `backend/ob_wiki/`, `backend/config.yaml`, `backend/.env`, `frontend/dist/`, `.venv/`, `node_modules/`, etc., are excluded by `.gitignore` and **will not** be distributed via git. See [Deployment Steps](#step-0-prepare-runtime-data-important).

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

# Option 2: Use repository script (Requires backend/.venv to exist and be activated)
./backend/run.sh
```

Health check:

```bash
curl -s http://127.0.0.1:8000/api/health
```

Expected output (Mock default, LLM unconfigured):

```json
{"status":"ok","ocp_provider":"mock","sql_provider":"mock","llm_configured":false,"memory_enabled":false}
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
| `ocp_tenants.json`, `ocp_clusters.json`, `topology.json` | `get_tenant_info` |
| `ocp_slow_sqls.json` (first `sqlId` is `sq-scan-orders-1`) | `get_slow_sql` |
| `ocp_sql_text.json`, `ocp_top_plan.json`, `ocp_sql_explain.json` | `get_full_sql_text`, `get_sql_top_plan`, `get_sql_explain` |
| `sample_tables.json` | `execute_sql`, plus the synthesized `SHOW CREATE TABLE` behind `get_table_ddl` |
| `slow_sqls.json` | the simplified `list_slow_sql` / `oceanbase.gv$sql_audit` mock path |
| `explain_results.json` | the mock `EXPLAIN` lookup |

With `ocp.provider: mock` **and** `sql_ro.provider: mock`, all seven tools answer from these fixtures, so the demo runs with neither OCP nor a database reachable. `real_*.json` files are for captured real responses and stay gitignored.

### Frontend

```bash
cd frontend
npm install        # First time
npm run dev        # http://127.0.0.1:5173 (/api is proxied → 127.0.0.1:8000)
```

Start the backend first as described above (defaults to mock mode), then start the frontend. Entering "有哪些慢SQL？" on the webpage exhibits two behaviors: when LLM is unconfigured, it follows the **503 error branch** (displaying "LLM not configured" at the top); after connecting a real or stub LLM, entering the same prompt demonstrates the complete demo loop: `status` grey text → markdown streaming → `done`.

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
| `meta_db` | `provider`                                               | `mock \| real`                                  |
| `meta_db` | `host`/`port`/`username`/`password`/`db_name`            | Direct connection to meta database when real (Unused in current version) |
| `sql_ro`  | `provider`                                               | `mock \| real`                                  |
| `sql_ro`  | `host_map`                                               | When real: Cluster name → Connection string mapping (dict string) |
| `sql_ro`  | `username`/`password`                                    | Read-only database account (SELECT privilege only recommended) |
| `sql_ro`  | `connect_timeout` / `query_timeout_seconds` / `max_rows` | Connection/query timeout and max row limits      |
| `llm`     | `base_url`/`api_key`/`model`                             | OpenAI-compatible model API (All three required to be considered "configured") |
| `llm`     | `temperature` / `max_input_tokens`                       | Sampling temperature / Context compression threshold benchmark |
| `agent`   | `send_row_data`                                          | Whether results passed to LLM contain row data  |
| `agent`   | `max_seconds`                                            | Fallback limit for total execution time of a single agent turn |
| `agent`   | `confirm_db_ops`                                         | Whether to enable human approval for `execute_sql` (HITL) |
| `agent`   | `confirm_timeout_seconds`                                | Approval timeout (Defaults to rejection on timeout) |
| `agent`   | `recursion_limit`                                        | Maximum recursion steps for langgraph           |
| `memory`  | `enabled`                                                | Persist conversations to PostgreSQL (`true` when PG is available) |
| `memory`  | `host`/`port`/`user`/`password`/`dbname`                 | PostgreSQL connection; set `dsn` to override the individual parts |
| `memory`  | `pool_min_size` / `pool_max_size`                        | Connection pool bounds (shared by checkpointer and history table) |
| `memory`  | `list_limit` / `messages_limit`                          | Default page caps for the thread-list / history endpoints |
| `auth`    | `enabled`                                                | Require `Authorization: Bearer <token>` on every `/api/*` route except `/api/health` |
| `auth`    | `token`                                                  | The shared token; `enabled: true` with an empty token fails fast at startup |

Environment variables with the same names use uppercase format (e.g., `OCP_PROVIDER`, `LLM_BASE_URL`, `SEND_ROW_DATA`, `MEMORY_ENABLED`, `MEMORY_DSN`, `MEMORY_HOST`, `MEMORY_PASSWORD`, `AUTH_ENABLED`, `AUTH_TOKEN`).

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
                 (ocp.base_url)    (meta_db / sql_ro)
```

### Step 0: Prepare Runtime Data (Important)

The following data **is not distributed via git** (excluded by `.gitignore`) and must be manually placed under the `backend/` working directory during deployment:

1. **`backend/config.yaml`** and **`backend/.env`**: Generate and populate with actual values according to [Configuration](#configuration).
2. **`backend/ob_wiki/`**: OceanBase official documentation knowledge base directory. The System Prompt expects the documentation entry point at `./ob_wiki/README.md`, which the agent references in read-only mode using file tools (restricted to this directory root). **Missing this directory will cause document retrieval features to fail**.
3. **PostgreSQL** (only when `memory.enabled: true`): a reachable instance plus a role allowed to create tables in `public`. The backend creates its own checkpoint and history tables on first start. See [Conversation Memory (PostgreSQL)](#conversation-memory-postgresql).

> Working directory convention: The backend runs with `backend/` as its working directory (`run.sh` will `cd` to the script directory), ensuring relative paths like `./ob_wiki` and `./config.yaml` work properly.

### Step 1: Backend Deployment

**A. Install dependencies and set up venv**

```bash
cd backend
python3.13 -m venv .venv            # Or use uv
source .venv/bin/activate
pip install -r requirements.txt      # Or: uv sync --project backend
```

In production, it is recommended to **disable reload** and run continuously.

**B. Start with Uvicorn**

```bash
cd /path/to/ob_agent/backend
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
```

- `--host 127.0.0.1`: The backend listens locally only; external reverse proxy (Nginx) exposes ports 443/80 externally.
- `--workers`: Adjust based on CPU cores and concurrency; for higher throughput, consider using gunicorn + uvicorn worker.
- Maintain `cd backend` throughout to ensure relative paths `./ob_wiki` and `./config.yaml` remain valid.

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
ExecStart=/opt/ob_agent/backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
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
- **LLM**: After configuration, change mock/demo prompt (system prompt rule 6) to inject per provider.
- **Oracle Tenant**: Connections currently marked as TODO (Not supported yet; `execute_sql` / `get_table_ddl` will display prompts).