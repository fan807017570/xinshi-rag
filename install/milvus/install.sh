#!/usr/bin/env bash
# =============================================================================
# install.sh — 启动 / 停止独立的 Milvus 基础设施
#
# 用法：
#   bash install/milvus/install.sh up       # 启动 etcd、MinIO、Milvus
#   bash install/milvus/install.sh down     # 停止并删除容器，保留数据卷
#   bash install/milvus/install.sh restart
#   bash install/milvus/install.sh status
#   bash install/milvus/install.sh logs [服务名]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"
ACTION="${1:-up}"

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
  echo "已创建默认配置：$ENV_FILE"
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "错误：未找到 Docker Compose，请先执行 install/install-docker.sh。" >&2
  exit 1
fi

compose() {
  "${COMPOSE[@]}" -p "${COMPOSE_PROJECT_NAME:-xinshi}" -f "$COMPOSE_FILE" "$@"
}

cd "$SCRIPT_DIR"
case "$ACTION" in
  up)
    compose up -d
    echo "Milvus 基础设施已启动。"
    echo "  Milvus: localhost:${MILVUS_EXPOSED_PORT:-19531}"
    echo "  MinIO : http://localhost:${MINIO_UI_PORT:-19001}  (minioadmin/minioadmin)"
    ;;
  down)
    compose down
    echo "Milvus 基础设施已停止，数据卷已保留。"
    ;;
  restart)
    compose restart
    ;;
  status)
    compose ps
    ;;
  logs)
    if [[ -n "${2:-}" ]]; then
      compose logs -f "$2"
    else
      compose logs -f
    fi
    ;;
  *)
    echo "用法：$0 {up|down|restart|status|logs [服务名]}" >&2
    exit 2
    ;;
esac
