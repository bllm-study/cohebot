#!/bin/bash
# cohebot 크롤링 실행 스크립트
# 사용법:
#   bash scripts/crawl.sh
#   bash scripts/crawl.sh --max-articles 5
#   DATA_DIR=data MAX_ARTICLES=10 bash scripts/crawl.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
MAX_ARTICLES="${MAX_ARTICLES:-}"

ARGS=("--data-dir" "$DATA_DIR")
if [ -n "$MAX_ARTICLES" ]; then
    ARGS+=("--max-articles" "$MAX_ARTICLES")
fi
ARGS+=("$@")

echo "=== 크롤링 시작 ==="
echo "uv run cohebot-crawl ${ARGS[*]}"
echo ""
uv run cohebot-crawl "${ARGS[@]}"
echo ""
echo "=== 크롤링 완료 ==="
