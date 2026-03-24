#!/bin/bash
# 올인원 GPU Cloud 학습 스크립트
#
# 사용법 (인스턴스에서):
#   curl -sSL https://raw.githubusercontent.com/bllm-study/cohebot/main/scripts/run.sh | bash
#
#   또는 이미 클론된 상태에서:
#   bash scripts/run.sh
#   bash scripts/run.sh --config configs/cloud.toml --batch-size 16
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_DIR="${WORKSPACE}/cohebot"

# --- 1단계: 환경 세팅 ---
if [ ! -d "$PROJECT_DIR/.git" ]; then
    echo ">>> 환경 세팅 시작..."

    apt-get update -qq && apt-get install -y -qq git curl > /dev/null 2>&1

    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    git clone https://github.com/bllm-study/cohebot.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    uv sync

    echo ">>> 환경 세팅 완료"
else
    cd "$PROJECT_DIR"
    echo ">>> 기존 환경 발견: ${PROJECT_DIR}"
fi

# --- 2단계: 학습 ---
echo ">>> 학습 시작..."

if [ $# -eq 0 ]; then
    bash scripts/train.sh --config configs/cloud.toml
else
    bash scripts/train.sh "$@"
fi
