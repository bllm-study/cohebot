import json
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from .wikipedia_cleaner import WikiTextCleaner


class WikipediaPreprocessor:
    """Wikipedia article preprocessor.

    Processes articles from Hugging Face dataset and builds corpus.
    """

    def __init__(
        self,
        data_dir: str = "data",
        min_text_length: int = 100,
        max_text_length: int = 100000,
    ):
        """Initialize preprocessor.

        Args:
            data_dir: Directory for data files.
            min_text_length: Minimum text length for validation.
            max_text_length: Maximum text length to keep.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = WikiTextCleaner(min_text_length)
        self.max_text_length = max_text_length

        self.clean_path = self.data_dir / "wiki_clean.jsonl"
        self.corpus_path = self.data_dir / "wiki_corpus.txt"

    def process_articles(
        self,
        articles: Iterator[dict]
    ) -> Iterator[dict]:
        """Process and validate articles.

        Args:
            articles: Iterator of article dictionaries.

        Yields:
            Cleaned article dictionaries with id, url, title, and text fields.
        """
        for article in articles:
            clean_text = self.cleaner.clean(article["text"])

            if len(clean_text) > self.max_text_length:
                clean_text = clean_text[:self.max_text_length]

            if not self.cleaner.is_valid(clean_text):
                continue

            yield {
                "id": article["id"],
                "url": article["url"],
                "title": article["title"],
                "text": clean_text
            }

    def save_clean_articles(self, articles: Iterator[dict]) -> int:
        """Save cleaned articles to JSONL file.

        Args:
            articles: Iterator of cleaned article dictionaries.

        Returns:
            Number of articles saved.
        """
        count = 0
        with open(self.clean_path, "w", encoding="utf-8") as f:
            for article in tqdm(articles, desc="정제 및 저장"):
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                count += 1
        print(f"저장 완료: {self.clean_path} ({count}개 문서)")
        return count

    def build_corpus(self, add_title: bool = True) -> int:
        """Build corpus from cleaned articles.

        Args:
            add_title: Whether to include article titles in corpus.

        Returns:
            Total number of characters in corpus.
        """
        if not self.clean_path.exists():
            raise FileNotFoundError(f"정제된 데이터가 없습니다: {self.clean_path}")

        total_chars = 0

        with open(self.clean_path, "r", encoding="utf-8") as fin:
            with open(self.corpus_path, "w", encoding="utf-8") as fout:
                for line in tqdm(fin, desc="코퍼스 생성"):
                    article = json.loads(line)

                    if add_title:
                        text = f"# {article['title']}\n\n{article['text']}"
                    else:
                        text = article["text"]

                    fout.write(text + "\n\n\n\n")
                    total_chars += len(text)

        print(f"코퍼스 생성 완료: {self.corpus_path}")
        print(f"총 {total_chars:,} 문자 ({total_chars / 1e6:.1f}MB)")

        return total_chars

    def load_corpus(self) -> str:
        """Load corpus from file.

        Returns:
            Corpus text content.

        Raises:
            FileNotFoundError: If corpus file does not exist.
        """
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"코퍼스가 없습니다: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_stats(self) -> dict:
        """Get statistics about processed data.

        Returns:
            Dictionary with dataset statistics.
        """
        stats = {}

        if self.clean_path.exists():
            with open(self.clean_path, "r", encoding="utf-8") as f:
                articles = [json.loads(line) for line in f]

            stats["num_articles"] = len(articles)
            stats["total_chars"] = sum(len(a["text"]) for a in articles)
            stats["avg_chars"] = stats["total_chars"] / max(len(articles), 1)

            lengths = [len(a["text"]) for a in articles]
            stats["min_length"] = min(lengths) if lengths else 0
            stats["max_length"] = max(lengths) if lengths else 0

        if self.corpus_path.exists():
            stats["corpus_size"] = self.corpus_path.stat().st_size

        return stats


def preprocess_wikipedia(
    data_dir: str = "data",
    max_articles: int | None = None,
) -> str:
    """Preprocess Wikipedia articles using Hugging Face crawler.

    Args:
        data_dir: Data directory.
        max_articles: Maximum number of articles to fetch.

    Returns:
        Path to generated corpus file.
    """
    from ..huggingface_crawler import HuggingFaceDatasetCrawler

    crawler = HuggingFaceDatasetCrawler(data_dir)
    articles = crawler.fetch_articles(max_articles)

    preprocessor = WikipediaPreprocessor(data_dir)
    clean_articles = preprocessor.process_articles(articles)
    preprocessor.save_clean_articles(clean_articles)
    preprocessor.build_corpus()

    stats = preprocessor.get_stats()
    print("\n=== 데이터셋 통계 ===")
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    return str(preprocessor.corpus_path)
