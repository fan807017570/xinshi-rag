#!/usr/bin/env bash
# =============================================================================
# install-deps.sh — 安装/检查 macOS 部署 xinshi-rag 所需的软件
#
# 用法：
#   bash install/macos/install-deps.sh --check
#   bash install/macos/install-deps.sh --install
#   bash install/macos/install-deps.sh --install --docker-only
#
# 说明：
#   - Docker 使用 Docker Desktop；脚本不会静默安装或替换已有 Docker 环境。
#   - 原生运行使用 Python 3.11，CPU 版 PyTorch 从 PyPI 安装。
# =============================================================================
set -euo pipefail

ACTION="check"
DOCKER_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --check)       ACTION="check" ;;
    --install)     ACTION="install" ;;
    --docker-only) DOCKER_ONLY=true ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "错误：未知参数 $arg" >&2
      echo "用法：$0 [--check|--install] [--docker-only]" >&2
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误：此脚本仅支持 macOS。" >&2
  exit 1
fi

HOST_ARCH="$(uname -m)"

has_command() {
  command -v "$1" >/dev/null 2>&1
}

BREW_BIN=""
if [[ "$HOST_ARCH" == "arm64" && -x /opt/homebrew/bin/brew ]]; then
  BREW_BIN="/opt/homebrew/bin/brew"
elif has_command brew; then
  BREW_BIN="$(command -v brew)"
fi

HAS_BREW=false
if [[ -n "$BREW_BIN" ]]; then
  HAS_BREW=true
elif [[ "$DOCKER_ONLY" == false ]]; then
  echo "错误：未找到 Homebrew。" >&2
  echo "请先从 https://brew.sh 安装 Homebrew，再重新执行本脚本。" >&2
  exit 1
fi

if [[ "$ACTION" == "install" ]]; then
  PYTHON_FORMULA_PREFIX=""
  if [[ "$HAS_BREW" == true ]]; then
    PYTHON_FORMULA_PREFIX="$($BREW_BIN --prefix python@3.11 2>/dev/null || true)"
  fi

  if [[ "$DOCKER_ONLY" == false && "$HAS_BREW" == true && ! -x "$PYTHON_FORMULA_PREFIX/bin/python3.11" ]]; then
    echo "==> 安装 Python 3.11"
    "$BREW_BIN" list --formula python@3.11 >/dev/null 2>&1 || "$BREW_BIN" install python@3.11
  fi

  if [[ "$HAS_BREW" == true ]] && ! has_command docker && [[ ! -d "/Applications/Docker.app" ]]; then
    echo "==> 安装 Docker Desktop"
    "$BREW_BIN" list --cask docker >/dev/null 2>&1 || "$BREW_BIN" install --cask docker
  fi

  if [[ -d "/Applications/Docker.app" ]] && ! docker info >/dev/null 2>&1; then
    echo "==> 启动 Docker Desktop"
    open -a Docker
    for i in $(seq 1 45); do
      if docker info >/dev/null 2>&1; then
        break
      fi
      sleep 2
    done
  fi
fi

if ! has_command docker; then
  echo "错误：未找到 docker 命令，请安装并启动 Docker Desktop。" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "错误：Docker Desktop 未运行，请先启动 Docker Desktop。" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "错误：未找到 Docker Compose Plugin，请升级 Docker Desktop。" >&2
  exit 1
fi

echo "Docker：$(docker --version)"
echo "Compose：$(docker compose version)"

if [[ "$DOCKER_ONLY" == false ]]; then
  PYTHON_BIN="${PYTHON_BIN:-}"
  if [[ -z "$PYTHON_BIN" ]]; then
    if [[ "$HOST_ARCH" == "arm64" && -x /opt/homebrew/opt/python@3.11/bin/python3.11 ]]; then
      PYTHON_BIN="/opt/homebrew/opt/python@3.11/bin/python3.11"
    elif has_command python3.11; then
      PYTHON_BIN="python3.11"
    elif has_command python3; then
      PYTHON_BIN="python3"
    elif [[ "$HAS_BREW" == true && -x "$($BREW_BIN --prefix python@3.11 2>/dev/null || true)/bin/python3.11" ]]; then
      PYTHON_BIN="$($BREW_BIN --prefix python@3.11)/bin/python3.11"
    fi
  fi

  if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误：未找到 Python 3，请执行：brew install python@3.11" >&2
    exit 1
  fi

  "$PYTHON_BIN" -c 'import sys; v=sys.version_info; sys.exit(0 if (v.major == 3 and v.minor >= 10) else 1)' || {
    echo "错误：Python 版本过低，需要 Python 3.10 或更高版本。" >&2
    exit 1
  }
  PYTHON_ARCH="$($PYTHON_BIN -c 'import platform; print(platform.machine())')"
  if [[ "$HOST_ARCH" == "arm64" && "$PYTHON_ARCH" != "arm64" ]]; then
    echo "错误：Apple Silicon 原生部署需要 arm64 Python，当前为 $PYTHON_ARCH：$PYTHON_BIN" >&2
    exit 1
  fi
  echo "Python：$($PYTHON_BIN --version 2>&1) ($PYTHON_ARCH)"
fi

echo "✅ macOS 部署依赖检查通过。"
