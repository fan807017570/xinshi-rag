#!/usr/bin/env bash
# =============================================================================
# install-ubuntu-deps.sh — 安装 Ubuntu 原生运行 xinshi-rag 所需的系统依赖
# =============================================================================
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "错误：当前用户不是 root，且未找到 sudo。" >&2
    exit 1
  fi
  SUDO=(sudo)
fi

if [[ ! -r /etc/os-release ]]; then
  echo "错误：无法识别当前操作系统。" >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "错误：此脚本仅支持 Ubuntu，当前系统为 ${ID:-unknown}。" >&2
  exit 1
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv

echo "Ubuntu 原生构建所需系统依赖已安装。"
