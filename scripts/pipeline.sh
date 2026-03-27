#!/bin/bash
# cohebot 크롤링 + 학습 통합 파이프라인
# 사용법:
#   bash scripts/pipeline.sh                          # 기본값 (crawl + train)
#   bash scripts/pipeline.sh --max-articles 5          # 소규모 테스트
#   MAX_ARTICLES=5 CONFIG=configs/test-cpu.toml bash scripts/pipeline.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
MAX_ARTICLES="${MAX_ARTICLES:-}"
CONFIG="${CONFIG:-configs/default.toml}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-articles) MAX_ARTICLES="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --data-dir)     DATA_DIR="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "=========================================="
echo "  CoheBot Pipeline"
echo "=========================================="
echo "  data-dir:     $DATA_DIR"
echo "  config:       $CONFIG"
echo "  max-articles: ${MAX_ARTICLES:-전체}"
echo "=========================================="
echo ""

# --- Phase 1: Crawl ---
if [ -f "${DATA_DIR}/wiki_corpus.txt" ]; then
    echo "[crawl] 기존 코퍼스 발견, 스킵: ${DATA_DIR}/wiki_corpus.txt"
else
    echo "[crawl] 위키백과 코퍼스 생성 중..."
    CRAWL_ARGS=("--data-dir" "$DATA_DIR")
    if [ -n "$MAX_ARTICLES" ]; then
        CRAWL_ARGS+=("--max-articles" "$MAX_ARTICLES")
    fi
    uv run cohebot-crawl "${CRAWL_ARGS[@]}" || true
    if [ ! -f "${DATA_DIR}/wiki_corpus.txt" ]; then
        echo "[crawl] ❌ 코퍼스 생성 실패" >&2
        exit 1
    fi
    echo ""
fi

# --- Phase 2: Train ---
echo "[train] 학습 시작 (config: $CONFIG)"
TRAIN_ARGS=("--config" "$CONFIG" "--data-dir" "$DATA_DIR" "${EXTRA_ARGS[@]}")
uv run cohebot-train "${TRAIN_ARGS[@]}"

echo ""
echo "=========================================="
echo "  Pipeline 완료 ✅"
echo "=========================================="
