from .base import BaseCrawler
from .huggingface_crawler import HuggingFaceDatasetCrawler
from .preprocessor.wikipedia_preprocessor import WikipediaPreprocessor, preprocess_wikipedia

__all__ = [
    "BaseCrawler",
    "HuggingFaceDatasetCrawler",
    "WikipediaPreprocessor",
    "preprocess_wikipedia",
    "main",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="한국어 위키피디아 코퍼스 생성 (HuggingFace streaming)")
    parser.add_argument("--data-dir", type=str, default="data", help="데이터 저장 디렉토리")
    parser.add_argument("--max-articles", type=int, default=None, help="최대 문서 수")
    parser.add_argument("--upload", action="store_true", help="HF Hub 업로드 방법 안내 출력")
    args = parser.parse_args()

    if args.upload:
        print("=== HuggingFace Hub 업로드 가이드 ===")
        print()
        print("1. 코퍼스 생성:")
        print(f"   uv run cohebot-crawl --data-dir {args.data_dir}")
        print()
        print("2. HuggingFace Hub에 업로드:")
        print(f"   uv run huggingface-cli upload bllm-study/kowiki-corpus {args.data_dir}/wiki_corpus.txt")
        print(f"   uv run huggingface-cli upload bllm-study/kowiki-corpus {args.data_dir}/wiki_clean.jsonl")
        print()
        print("3. (선택) README 생성 후 업로드:")
        print("   uv run huggingface-cli upload bllm-study/kowiki-corpus README.md")
        print()
        return

    preprocess_wikipedia(data_dir=args.data_dir, max_articles=args.max_articles)
