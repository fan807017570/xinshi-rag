#!/usr/bin/env bash
# =============================================================================
# deploy-components.sh — macOS 下部署 Milvus、etcd 和 MinIO 基础组件
#
# 用法：
#   bash install/macos/deploy-components.sh up
#   bash install/macos/deploy-components.sh status
#   bash install/macos/deploy-components.sh logs [服务名]
#   bash install/macos/deploy-components.sh restart
#   bash install/macos/deploy-components.sh down
#
# Apple Silicon 默认通过 Docker Desktop 的 amd64 兼容层运行组件。
# 可用 DOCKER_PLATFORM=linux/arm64 覆盖，但前提是所用镜像支持该架构。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
ACTION="${1:-up}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：此脚本仅支持 macOS。" >&2
  exit 1
fi

case "$ACTION" in
  up|down|restart|status|logs) ;;
  -h|--help)
    sed -n '2,16p' "$0"
    exit 0
    ;;
  *)
    echo "错误：未知操作 $ACTION" >&2
    exit 2
    ;;
esac

bash "$SCRIPT_DIR/install-deps.sh" --check --docker-only

# Milvus 2.4.4 在 Apple Silicon 上优先使用 amd64 兼容模式。
export DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
export DOCKER_DEFAULT_PLATFORM="${DOCKER_DEFAULT_PLATFORM:-$DOCKER_PLATFORM}"

echo "==> Docker 平台：$DOCKER_PLATFORM"

if [[ "$ACTION" != "up" ]]; then
  exec bash "$PROJECT_ROOT/install/milvus/install.sh" "$@"
fi

bash "$PROJECT_ROOT/install/milvus/install.sh" "$@"

echo "==> 等待 Milvus 健康检查通过"
for i in $(seq 1 36); do
  status="$(docker inspect xinshi-milvus --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)"
  echo "   [$i/36] $status"
  if [[ "$status" == "healthy" ]]; then
    echo "✅ Milvus、etcd 和 MinIO 已就绪。"
    exit 0
  fi
  if [[ "$status" == "unhealthy" ]]; then
    echo "错误：Milvus 健康检查失败，请查看日志：" >&2
    echo "  bash $0 logs milvus-standalone" >&2
    exit 1
  fi
  sleep 5
done

echo "错误：等待 Milvus 就绪超时，请查看日志：" >&2
echo "  bash $0 logs milvus-standalone" >&2
exit 1
