#!/usr/bin/env bash
# 启动后端（工作目录固定为本脚本所在目录，保证 ./config.yaml、./doc 等相对路径生效）
#
# 依赖来源：backend/requirements.txt —— 仓库不提供 pyproject.toml / uv.lock，
# 因此不再使用 uv sync。
#   - 没有 .venv：自动创建，并从 requirements.txt 安装依赖
#   - 已有 .venv：直接启动；仅当关键依赖缺失时给出明确的安装命令（不自动升级，
#     避免每次启动都装包）
# 需要指定解释器时：PYTHON_BIN=python3.13 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
PY="$VENV/bin/python"

# 关键顶层依赖：覆盖 requirements.txt 里的运行期必需包（含最新加入的检查点依赖），
# 用于判断既有 venv 能否直接跑起来。
REQUIRED_IMPORTS="fastapi, uvicorn, langchain, langgraph, psycopg, psycopg_pool, langgraph.checkpoint.postgres"

if [[ ! -x "$PY" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-}"
  if [[ -z "$PYTHON_BIN" ]]; then
    for cand in python3.13 python3.12 python3; do
      if command -v "$cand" >/dev/null 2>&1; then PYTHON_BIN="$cand"; break; fi
    done
  fi
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "未找到可用的 python3（需要 Python >= 3.13）；可用 PYTHON_BIN=/path/to/python3.13 $0 指定。" >&2
    exit 1
  fi

  echo "[run.sh] 未找到 $VENV，正在用 $PYTHON_BIN 创建 …" >&2
  if ! "$PYTHON_BIN" -m venv "$VENV"; then
    rm -rf "$VENV"   # 清掉半成品，保证下次运行能重新创建
    echo "[run.sh] 创建 venv 失败。若上面提示 ensurepip 不可用，说明该解释器缺少 venv 支持：" >&2
    echo "    - Debian/Ubuntu：sudo apt install python3-venv" >&2
    echo "    - 或改用 uv：uv venv $VENV && uv pip install --python $PY -r requirements.txt" >&2
    echo "    - 或指定其他解释器：PYTHON_BIN=/path/to/python3.13 $0" >&2
    exit 1
  fi
  echo "[run.sh] 按 requirements.txt 安装依赖 …" >&2
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
elif ! "$PY" -c "import $REQUIRED_IMPORTS" >/dev/null 2>&1; then
  echo "[run.sh] $VENV 中缺少运行期依赖（例如刚拉取到新增依赖的代码）。请先安装：" >&2
  echo "    $PY -m pip install -r requirements.txt" >&2
  echo "  若该 venv 由 uv 创建（内部没有 pip），改用：" >&2
  echo "    uv pip install --python $PY -r requirements.txt" >&2
  exit 1
fi

# 直接用 venv 解释器而非 activate，避免依赖 PATH 改动
exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
