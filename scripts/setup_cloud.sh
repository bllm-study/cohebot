#!/bin/bash
# GPU Cloud 인스턴스 환경 세팅
# RunPod / Hyperbolic 등에서 최초 1회 실행
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bllm-study/cohebot.git}"
WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_DIR="${WORKSPACE}/cohebot"

echo "=== cohebot 환경 세팅 ==="
echo "  워크스페이스: ${WORKSPACE}"

# --- 기본 패키지 ---
apt-get update -qq && apt-get install -y -qq git curl > /dev/null 2>&1
echo "[1/4] 기본 패키지 설치 완료"

# --- uv 설치 ---
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[2/4] uv 설치 완료 ($(uv --version))"

# --- 레포 클론 ---
if [ -d "$PROJECT_DIR" ]; then
    echo "[3/4] 기존 레포 발견, pull..."
    cd "$PROJECT_DIR"
    git pull --ff-only
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi
echo "[3/4] 레포 준비 완료"

# --- 의존성 설치 ---
uv sync
echo "[4/4] 의존성 설치 완료"

# --- GPU 정보 출력 ---
echo ""
echo "=== GPU 정보 ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU 미감지"

echo ""
echo "=== 세팅 완료 ==="
echo "학습 시작: bash scripts/train.sh --config configs/cloud.toml"
