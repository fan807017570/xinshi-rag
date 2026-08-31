#!/usr/bin/env bash
# =============================================================================
# build-ubuntu.sh — 为 Ubuntu 原生运行方式创建虚拟环境并安装依赖
#
# 该脚本只构建应用运行环境，不启动服务，也不安装 Milvus。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f /etc/os-release ]]; then
  echo "错误：此脚本仅支持 Ubuntu。" >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "错误：此脚本仅支持 Ubuntu，当前系统为 ${ID:-unknown}。" >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "错误：未找到 $PYTHON_BIN，请先执行 install/install-ubuntu-deps.sh。" >&2
  exit 1
fi

echo "==> 创建 Python 虚拟环境：$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "==> 升级 pip 工具"
"$VENV_PYTHON" -m pip install --upgrade pip "setuptools>=70.0.0,<81.0.0" wheel

echo "==> 安装 Ubuntu CPU 依赖"
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

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

echo "✅ Ubuntu 原生构建完成"
echo "   启动命令：bash $SCRIPT_DIR/run-ubuntu.sh"
