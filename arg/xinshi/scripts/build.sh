#!/usr/bin/env bash
# =============================================================================
# build.sh — 一键构建 xinshi-rag Docker 镜像
#
# 用法：
#   bash scripts/build.sh              # 构建，tag = xinshi-rag:latest
#   bash scripts/build.sh 1.2.0        # 构建，tag = xinshi-rag:1.2.0 + latest
#   bash scripts/build.sh 1.2.0 --push # 构建并推送到镜像仓库
#
# 环境变量（可在 .env 中设置，或启动前 export）：
#   REGISTRY   镜像仓库前缀，默认为空（本地镜像）
#              示例：export REGISTRY=registry.example.com/xinshi
# =============================================================================
set -euo pipefail

# ── 路径定位 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XINSHI_DIR="$(dirname "$SCRIPT_DIR")"          # arg/xinshi/
PROJECT_ROOT="$(dirname "$(dirname "$XINSHI_DIR")")"  # xinshi-rag/

# ── 参数解析 ─────────────────────────────────────────────────────────────────
VERSION="${1:-}"
PUSH=false
for arg in "$@"; do
  [[ "$arg" == "--push" ]] && PUSH=true
done

# ── 镜像名 ───────────────────────────────────────────────────────────────────
REGISTRY="${REGISTRY:-}"
BASE_NAME="xinshi-rag"
[[ -n "$REGISTRY" ]] && BASE_NAME="${REGISTRY%/}/${BASE_NAME}"

TAGS=("${BASE_NAME}:latest")
[[ -n "$VERSION" && "$VERSION" != "--push" ]] && TAGS+=("${BASE_NAME}:${VERSION}")

# ── 构建参数 ─────────────────────────────────────────────────────────────────
TAG_ARGS=()
for t in "${TAGS[@]}"; do
  TAG_ARGS+=("-t" "$t")
done

# ── 打印摘要 ─────────────────────────────────────────────────────────────────
echo "========================================"
echo "  xinshi-rag 镜像构建"
echo "========================================"
echo "  构建上下文 : $PROJECT_ROOT"
echo "  Dockerfile : $XINSHI_DIR/Dockerfile"
echo "  Tags       : ${TAGS[*]}"
echo "  推送       : $PUSH"
echo "========================================"
echo ""

# ── 构建 ─────────────────────────────────────────────────────────────────────
docker build \
  "${TAG_ARGS[@]}" \
  -f "$XINSHI_DIR/Dockerfile" \
  "$PROJECT_ROOT"

echo ""
echo "✅ 构建完成：${TAGS[*]}"

# ── 推送（可选）──────────────────────────────────────────────────────────────
if [[ "$PUSH" == true ]]; then
  for t in "${TAGS[@]}"; do
    echo "📤 推送 $t ..."
    docker push "$t"
  done
  echo "✅ 推送完成"
fi
