#!/usr/bin/env bash
# =============================================================================
# deploy.sh — 一键启动 / 重启 xinshi-rag Docker 应用
#
# Milvus、MinIO 和 etcd 由 install/milvus/install.sh 独立管理。
#
# 用法：
#   bash rag/docker-install/deploy.sh            # 启动（不重新构建镜像）
#   bash rag/docker-install/deploy.sh --build    # 先重新构建镜像再启动
#   bash rag/docker-install/deploy.sh --down     # 停止并删除容器（数据卷保留）
#   bash rag/docker-install/deploy.sh --restart  # 重启所有容器
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
APP_DIR="$PROJECT_ROOT/rag"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$APP_DIR/.env"

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "错误：未找到 Docker Compose，请先执行 install/install-docker.sh。" >&2
  exit 1
fi

compose() {
  "${COMPOSE[@]}" --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

MODE="up"
BUILD=false
for arg in "$@"; do
  case "$arg" in
    --build)   BUILD=true ;;
    --down)    MODE="down" ;;
    --restart) MODE="restart" ;;
  esac
done

# ── 检查 .env ─────────────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo "⚠️  未找到 $ENV_FILE"
  echo "   请先复制模板并填写 DEEPSEEK_API_KEY："
  echo "   cp $APP_DIR/.env.example $ENV_FILE"
  exit 1
fi

# 将 .env 中的网络名和端口提供给脚本本身；Compose 也会读取同一个文件。
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
NETWORK_NAME="${XINSHI_NETWORK:-xinshi-net}"

if ! grep -q "DEEPSEEK_API_KEY=sk-" "$ENV_FILE" 2>/dev/null; then
  echo "⚠️  $ENV_FILE 中 DEEPSEEK_API_KEY 未填写或格式不正确"
  echo "   请编辑 $ENV_FILE 填入真实 Key"
  exit 1
fi

# ── 执行 ─────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

case "$MODE" in
  down)
    echo "🛑 停止并删除容器..."
    compose down
    echo "✅ 已停止（数据卷已保留）"
    ;;
  restart)
    echo "🔄 重启所有容器..."
    compose restart
    echo "✅ 重启完成"
    ;;
  up)
    if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
      echo "错误：未找到共享网络 $NETWORK_NAME。" >&2
      echo "请先启动 Milvus 基础设施：" >&2
      echo "  bash $PROJECT_ROOT/install/milvus/install.sh up" >&2
      exit 1
    fi
    if [[ "$BUILD" == true ]]; then
      echo "🔨 重新构建镜像..."
      bash "$SCRIPT_DIR/build.sh"
    fi
    echo "🚀 启动服务栈..."
    compose up -d

    echo ""
    echo "⏳ 等待 Milvus 就绪..."
    for i in $(seq 1 24); do
      status=$(docker inspect xinshi-milvus --format "{{.State.Health.Status}}" 2>/dev/null || echo "unknown")
      printf "  [%02d] %s\n" "$i" "$status"
      [[ "$status" == "healthy" ]] && break
      sleep 5
    done

    echo ""
    echo "========================================"
    echo "  服务地址"
    echo "========================================"

    RAG_PORT="${RAG_PORT:-18765}"
    echo "  RAG 服务  : http://localhost:${RAG_PORT}"

    echo "  Milvus    : 由 install/milvus 独立管理"
    echo "========================================"
    ;;
esac
