#!/usr/bin/env bash
# =============================================================================
# deploy-macos.sh — macOS 下启动/停止 xinshi-rag Docker 应用
#
# 用法与 deploy.sh 相同：
#   bash rag/docker-install/deploy-macos.sh
#   bash rag/docker-install/deploy-macos.sh --build
#   bash rag/docker-install/deploy-macos.sh --restart
#   bash rag/docker-install/deploy-macos.sh --down
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：此脚本仅支持 macOS。" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "错误：未找到 docker，请先执行 bash install/macos/install-deps.sh --install --docker-only" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "错误：Docker Desktop 未运行。" >&2
  exit 1
fi

export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
export DOCKER_DEFAULT_PLATFORM="${DOCKER_DEFAULT_PLATFORM:-$DOCKER_PLATFORM}"
echo "==> Docker 平台：$DOCKER_PLATFORM"

exec bash "$SCRIPT_DIR/deploy.sh" "$@"
