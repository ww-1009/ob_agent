#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then
  echo "未找到 backend/.venv，请先运行: uv sync --project backend --extra dev" >&2
  exit 1
fi
source .venv/bin/activate
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
