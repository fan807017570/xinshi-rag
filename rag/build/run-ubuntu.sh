#!/usr/bin/env bash
# =============================================================================
# run-ubuntu.sh — 在 Ubuntu 上直接启动 xinshi-rag（不经过 Docker）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
APP_DIR="$PROJECT_ROOT/rag"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "错误：未找到虚拟环境 $VENV_DIR，请先执行：" >&2
  echo "  bash $SCRIPT_DIR/build-ubuntu.sh" >&2
  exit 1
fi

if [[ ! -f "$APP_DIR/.env" ]]; then
  echo "提示：未找到 $APP_DIR/.env，请先复制 .env.example 并填写配置。" >&2
else
  set -a
  # shellcheck disable=SC1090
  source "$APP_DIR/.env"
  set +a
fi

cd "$PROJECT_ROOT"
exec "$VENV_DIR/bin/python" -m uvicorn rag.server:app \
  --host "${XINSHI_HOST:-0.0.0.0}" \
  --port "${XINSHI_PORT:-${RAG_PORT:-8765}}"
