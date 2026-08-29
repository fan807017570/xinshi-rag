#!/usr/bin/env bash
# =============================================================================
# deploy.sh — 一键启动 / 重启完整服务栈（RAG + Milvus + MinIO + etcd）
#
# 用法：
#   bash scripts/deploy.sh            # 启动（不重新构建镜像）
#   bash scripts/deploy.sh --build    # 先重新构建镜像再启动
#   bash scripts/deploy.sh --down     # 停止并删除容器（数据卷保留）
#   bash scripts/deploy.sh --restart  # 重启所有容器
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XINSHI_DIR="$(dirname "$SCRIPT_DIR")"   # arg/xinshi/，docker-compose.yml 所在目录

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
ENV_FILE="$XINSHI_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "⚠️  未找到 $ENV_FILE"
  echo "   请先复制模板并填写 DEEPSEEK_API_KEY："
  echo "   cp $XINSHI_DIR/.env.example $ENV_FILE"
  exit 1
fi

if ! grep -q "DEEPSEEK_API_KEY=sk-" "$ENV_FILE" 2>/dev/null; then
  echo "⚠️  $ENV_FILE 中 DEEPSEEK_API_KEY 未填写或格式不正确"
  echo "   请编辑 $ENV_FILE 填入真实 Key"
  exit 1
fi

# ── 执行 ─────────────────────────────────────────────────────────────────────
cd "$XINSHI_DIR"

case "$MODE" in
  down)
    echo "🛑 停止并删除容器..."
    docker-compose down
    echo "✅ 已停止（数据卷已保留）"
    ;;
  restart)
    echo "🔄 重启所有容器..."
    docker-compose restart
    echo "✅ 重启完成"
    ;;
  up)
    if [[ "$BUILD" == true ]]; then
      echo "🔨 重新构建镜像..."
      bash "$SCRIPT_DIR/build.sh"
    fi
    echo "🚀 启动服务栈..."
    docker-compose up -d

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

    RAG_PORT=$(grep "^RAG_PORT=" "$ENV_FILE" | cut -d= -f2 || echo "18765")
    RAG_PORT="${RAG_PORT:-18765}"
    echo "  RAG 服务  : http://localhost:${RAG_PORT}"

    MILVUS_PORT=$(grep "^MILVUS_EXPOSED_PORT=" "$ENV_FILE" | cut -d= -f2 || echo "19531")
    MILVUS_PORT="${MILVUS_PORT:-19531}"
    echo "  Milvus    : localhost:${MILVUS_PORT}"

    MINIO_UI=$(grep "^MINIO_UI_PORT=" "$ENV_FILE" | cut -d= -f2 || echo "19001")
    MINIO_UI="${MINIO_UI:-19001}"
    echo "  MinIO UI  : http://localhost:${MINIO_UI}  (admin/minioadmin)"
    echo "========================================"
    ;;
esac
