#!/usr/bin/env bash
# =============================================================================
# install-docker.sh — 在 Ubuntu 上安装 Docker Engine、Buildx 和 Compose Plugin
#
# 用法：
#   bash install/install-docker.sh
#
# 脚本可重复执行。安装完成后，重新登录或执行 `newgrp docker`，
# 让当前用户可以不加 sudo 使用 docker 命令。
# =============================================================================
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=()
  INSTALL_USER="${SUDO_USER:-}"
else
  if ! command -v sudo >/dev/null 2>&1; then
    echo "错误：当前用户不是 root，且未找到 sudo。" >&2
    exit 1
  fi
  SUDO=(sudo)
  INSTALL_USER="${USER}"
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

ARCH="$(dpkg --print-architecture)"
CODENAME="${VERSION_CODENAME:-}"
if [[ -z "$CODENAME" ]]; then
  CODENAME="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-}")"
fi
if [[ -z "$CODENAME" ]]; then
  echo "错误：无法识别 Ubuntu 版本代号。" >&2
  exit 1
fi

echo "==> 安装 Docker 官方软件源前置依赖"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y ca-certificates curl

echo "==> 配置 Docker 官方软件源"
"${SUDO[@]}" install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | "${SUDO[@]}" gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
"${SUDO[@]}" chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
  | "${SUDO[@]}" tee /etc/apt/sources.list.d/docker.list >/dev/null

echo "==> 安装 Docker Engine、Buildx 和 Compose Plugin"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
"${SUDO[@]}" systemctl enable --now docker

if [[ -n "$INSTALL_USER" && "$INSTALL_USER" != "root" ]]; then
  "${SUDO[@]}" usermod -aG docker "$INSTALL_USER"
  echo "==> 已将 ${INSTALL_USER} 加入 docker 用户组"
  echo "    请重新登录，或执行：newgrp docker"
fi

docker_version=("${SUDO[@]}" docker --version)
compose_version=("${SUDO[@]}" docker compose version)
echo "==> 安装完成"
"${docker_version[@]}"
"${compose_version[@]}"
