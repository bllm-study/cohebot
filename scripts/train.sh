#!/bin/bash
# cohebot 학습 실행 스크립트
# 사용법:
#   bash scripts/train.sh --config configs/cloud.toml
#   bash scripts/train.sh --config configs/cloud.toml --batch-size 16
#   CONFIG=configs/cloud.toml bash scripts/train.sh
#   NUM_GPUS=4 bash scripts/train.sh --config configs/cloud.toml  # Multi-GPU
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
PROJECT_DIR="${WORKSPACE}/cohebot"

# 프로젝트 루트로 이동
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
fi

# --- GPU 확인 ---
echo "=== GPU 상태 ==="
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || echo "GPU 미감지 (CPU 모드)"
echo ""

# --- 데이터 확인 ---
DATA_DIR="${DATA_DIR:-data}"
if [ ! -f "${DATA_DIR}/wiki_corpus.txt" ]; then
    echo "코퍼스 파일 없음. HuggingFace Hub에서 다운로드 중..."
    uv run huggingface-cli download bllm-study/kowiki-corpus --local-dir "$DATA_DIR"
fi

# --- 학습 인자 구성 ---
TRAIN_ARGS=("$@")
if [ -n "${CONFIG:-}" ]; then
    HAS_CONFIG=false
    for arg in "$@"; do
        if [[ "$arg" == "--config" ]]; then
            HAS_CONFIG=true
            break
        fi
    done
    if [ "$HAS_CONFIG" = false ]; then
        TRAIN_ARGS=(--config "$CONFIG" "$@")
    fi
fi

# --- 학습 실행 ---
NUM_GPUS="${NUM_GPUS:-1}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "=== Multi-GPU 학습 시작 (${NUM_GPUS} GPUs) ==="
    echo "uv run torchrun --nproc_per_node=$NUM_GPUS -m cohebot.train ${TRAIN_ARGS[*]}"
    echo ""
    uv run torchrun --nproc_per_node="$NUM_GPUS" -m cohebot.train "${TRAIN_ARGS[@]}"
else
    echo "=== 학습 시작 ==="
    echo "uv run cohebot-train ${TRAIN_ARGS[*]}"
    echo ""
    uv run cohebot-train "${TRAIN_ARGS[@]}"
fi

echo ""
echo "=== 학습 완료 ==="
