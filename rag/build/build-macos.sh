#!/usr/bin/env bash
# =============================================================================
# build-macos.sh — 为 macOS 原生运行方式创建虚拟环境并安装依赖
#
# 该脚本只构建应用运行环境，不启动服务，也不安装 Milvus。
# Milvus、etcd 和 MinIO 仍通过 install/macos/deploy-components.sh 运行。
#
# 用法：
#   bash rag/build/build-macos.sh
#   PYTHON_BIN=python3.11 VENV_DIR=/path/to/venv bash rag/build/build-macos.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
HOST_ARCH="$(uname -m)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：此脚本仅支持 macOS。" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ "$HOST_ARCH" == "arm64" && -x /opt/homebrew/opt/python@3.11/bin/python3.11 ]]; then
    PYTHON_BIN="/opt/homebrew/opt/python@3.11/bin/python3.11"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v brew >/dev/null 2>&1 && [[ -x "$(brew --prefix python@3.11 2>/dev/null || true)/bin/python3.11" ]]; then
    PYTHON_BIN="$(brew --prefix python@3.11)/bin/python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "错误：未找到 $PYTHON_BIN，请先执行：brew install python@3.11" >&2
  exit 1
fi

"$PYTHON_BIN" -c 'import sys; v=sys.version_info; sys.exit(0 if (v.major == 3 and v.minor >= 10) else 1)' || {
  echo "错误：Python 版本过低，需要 Python 3.10 或更高版本。" >&2
  exit 1
}

PYTHON_ARCH="$($PYTHON_BIN -c 'import platform; print(platform.machine())')"
echo "==> Python：$($PYTHON_BIN --version 2>&1) ($PYTHON_ARCH)"

if [[ "$HOST_ARCH" == "arm64" && "$PYTHON_ARCH" != "arm64" ]]; then
  echo "错误：Apple Silicon 原生部署不能使用 Intel Python：$PYTHON_BIN" >&2
  echo "请使用：PYTHON_BIN=/opt/homebrew/opt/python@3.11/bin/python3.11 bash $0" >&2
  exit 1
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  EXISTING_VENV_ARCH="$($VENV_DIR/bin/python -c 'import platform; print(platform.machine())')"
  if [[ "$EXISTING_VENV_ARCH" != "$PYTHON_ARCH" ]]; then
    VENV_BACKUP="${VENV_DIR}.${EXISTING_VENV_ARCH}.backup"
    if [[ -e "$VENV_BACKUP" ]]; then
      echo "错误：虚拟环境备份已存在：$VENV_BACKUP" >&2
      echo "请确认备份内容后移走该目录，再重新执行构建。" >&2
      exit 1
    fi
    echo "==> 现有虚拟环境架构为 $EXISTING_VENV_ARCH，备份到：$VENV_BACKUP"
    mv "$VENV_DIR" "$VENV_BACKUP"
  fi
fi

echo "==> 创建 Python 虚拟环境：$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "==> 升级 pip 工具"
"$VENV_PYTHON" -m pip install --upgrade pip "setuptools>=70.0.0,<81.0.0" wheel

echo "==> 安装 cryptography 二进制 wheel"
"$VENV_PYTHON" -m pip install --only-binary=:all: "cryptography>=42.0.0"

echo "==> 安装 macOS 依赖"
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-macos.txt"

echo "==> 检查关键依赖"
"$VENV_PYTHON" - <<'PY'
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pymilvus\.client",
)

import fastapi
import langchain_deepseek
import langchain_huggingface
import pymilvus
import sentence_transformers
import torch
import transformers

print("fastapi/langchain/pymilvus/sentence-transformers/torch/transformers: ok")
print(f"torch={torch.__version__}")
PY

echo "✅ macOS 原生构建完成"
echo "   启动命令：bash $SCRIPT_DIR/run-macos.sh"
