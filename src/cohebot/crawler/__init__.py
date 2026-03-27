from .base import BaseCrawler
from .huggingface_crawler import HuggingFaceDatasetCrawler
from .preprocessor.wikipedia_preprocessor import WikipediaPreprocessor

__all__ = [
    "BaseCrawler",
    "HuggingFaceDatasetCrawler",
    "WikipediaPreprocessor",
    "preprocess_wikipedia",
    "main",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="한국어 위키피디아 크롤러 (HuggingFace)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="데이터 저장 디렉토리",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="최대 문서 수",
    )
    args = parser.parse_args()

    crawler = HuggingFaceDatasetCrawler(data_dir=args.data_dir)
    crawler.prepare_data()

    count = 0
    for article in crawler.fetch_articles(max_articles=args.max_articles):
        count += 1
        if count % 1000 == 0:
            print(f"  {count}개 문서 수집됨...")

    print(f"\n총 {count}개 문서 수집 완료")
