#!/usr/bin/env bash
# =============================================================================
# run-macos.sh — 在 macOS 上直接启动 xinshi-rag（不经过应用 Docker）
#
# 启动前请先用 install/macos/deploy-components.sh up 启动 Milvus 基础组件。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
APP_DIR="$PROJECT_ROOT/rag"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
ENV_FILE="$APP_DIR/.env"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：此脚本仅支持 macOS。" >&2
  exit 1
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "错误：未找到虚拟环境 $VENV_DIR，请先执行：" >&2
  echo "  bash $SCRIPT_DIR/build-macos.sh" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "提示：未找到 $ENV_FILE，将使用脚本默认值；生产环境请配置 API Key。" >&2
fi

cd "$PROJECT_ROOT"

echo "🚀 启动 xinshi-rag（原生 macOS）"
echo "   地址：http://127.0.0.1:${XINSHI_PORT:-${RAG_PORT:-8765}}"
echo "   配置：${XINSHI_CONFIG_FILE:-$PROJECT_ROOT/config/application.ini}"
exec "$VENV_DIR/bin/python" -m uvicorn rag.server:app \
  --host "${XINSHI_HOST:-0.0.0.0}" \
  --port "${XINSHI_PORT:-${RAG_PORT:-8765}}"
