import re
import json
import unicodedata
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False
    print("경고: mwparserfromhell 미설치. 기본 정규식 사용.")


class WikiTextCleaner:

    WIKI_PATTERNS = [
        (r"\{\{[^}]*\}\}", ""),
        (r"\[\[(분류|Category|파일|File|Image):[^\]]*\]\]", ""),
        (r"\[\[[^\]]*\|([^\]]*)\]\]", r"\1"),
        (r"\[\[([^\]]*)\]\]", r"\1"),
        (r"\[https?://[^\s\]]*\s*([^\]]*)\]", r"\1"),
        (r"\[https?://[^\]]*\]", ""),
        (r"<ref[^>]*>.*?</ref>", ""),
        (r"<ref[^>]*/>", ""),
        (r"<!--.*?-->", ""),
        (r"<[^>]+>", ""),
        (r"\{\|.*?\|\}", ""),
        (r"'{2,5}", ""),
        (r"^=+\s*(.+?)\s*=+$", r"\1"),
        (r"^[\*#:;]+\s*", ""),
        (r"__[A-Z]+__", ""),
        (r"\{\{|\}\}", ""),
    ]

    NORMALIZE_PATTERNS = [
        (r"[ \t]+", " "),
        (r"\n{3,}", "\n\n"),
        (r"^[ \t]+|[ \t]+$", ""),
    ]

    def __init__(self, min_text_length: int = 100):
        self.min_text_length = min_text_length

        self.wiki_patterns = [
            (re.compile(p, re.MULTILINE | re.DOTALL), r)
            for p, r in self.WIKI_PATTERNS
        ]
        self.normalize_patterns = [
            (re.compile(p, re.MULTILINE), r)
            for p, r in self.NORMALIZE_PATTERNS
        ]

    def clean_wiki_markup(self, text: str) -> str:
        if HAS_MWPARSER:
            try:
                parsed = mwparserfromhell.parse(text)
                text = parsed.strip_code(
                    normalize=True,
                    collapse=True,
                    keep_template_params=False
                )
            except Exception:
                pass

        for pattern, replacement in self.wiki_patterns:
            text = pattern.sub(replacement, text)

        return text

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)

        for pattern, replacement in self.normalize_patterns:
            text = pattern.sub(replacement, text)

        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    def clean(self, text: str) -> str:
        text = self.clean_wiki_markup(text)
        text = self.normalize_text(text)
        return text.strip()

    def is_valid(self, text: str) -> bool:
        if len(text) < self.min_text_length:
            return False

        korean_chars = len(re.findall(r"[가-힣]", text))
        if korean_chars / max(len(text), 1) < 0.3:
            return False

        return True


class WikipediaPreprocessor:

    def __init__(
        self,
        data_dir: str = "data",
        min_text_length: int = 100,
        max_text_length: int = 100000,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = WikiTextCleaner(min_text_length)
        self.max_text_length = max_text_length

        self.raw_path = self.data_dir / "wiki_raw.jsonl"
        self.clean_path = self.data_dir / "wiki_clean.jsonl"
        self.corpus_path = self.data_dir / "wiki_corpus.txt"

    def process_articles(
        self,
        articles: Iterator[dict],
        save_raw: bool = True
    ) -> Iterator[dict]:
        raw_file = None
        if save_raw:
            raw_file = open(self.raw_path, "w", encoding="utf-8")

        try:
            for article in articles:
                if raw_file:
                    raw_file.write(json.dumps(article, ensure_ascii=False) + "\n")

                clean_text = self.cleaner.clean(article["text"])

                if len(clean_text) > self.max_text_length:
                    clean_text = clean_text[:self.max_text_length]

                if not self.cleaner.is_valid(clean_text):
                    continue

                yield {
                    "title": article["title"],
                    "text": clean_text
                }
        finally:
            if raw_file:
                raw_file.close()

    def save_clean_articles(self, articles: Iterator[dict]) -> int:
        count = 0
        with open(self.clean_path, "w", encoding="utf-8") as f:
            for article in tqdm(articles, desc="정제 및 저장"):
                f.write(json.dumps(article, ensure_ascii=False) + "\n")
                count += 1
        print(f"저장 완료: {self.clean_path} ({count}개 문서)")
        return count

    def build_corpus(self, add_title: bool = True) -> int:
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

                    fout.write(text + "\n\n<|endoftext|>\n\n")
                    total_chars += len(text)

        print(f"코퍼스 생성 완료: {self.corpus_path}")
        print(f"총 {total_chars:,} 문자 ({total_chars / 1e6:.1f}MB)")

        return total_chars

    def load_corpus(self) -> str:
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"코퍼스가 없습니다: {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            return f.read()

    def get_stats(self) -> dict:
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
    use_api: bool = False,
) -> str:
    from .crawler import WikipediaDumpCrawler, WikipediaAPICrawler

    if use_api:
        crawler = WikipediaAPICrawler(data_dir)
        articles = crawler.get_random_articles(max_articles or 1000)
    else:
        crawler = WikipediaDumpCrawler(data_dir)
        crawler.download_dump()
        articles = crawler.parse_dump(max_articles)

    preprocessor = WikipediaPreprocessor(data_dir)

    clean_articles = preprocessor.process_articles(articles)
    preprocessor.save_clean_articles(clean_articles)

    preprocessor.build_corpus()

    stats = preprocessor.get_stats()
    print("\n=== 데이터셋 통계 ===")
    for key, value in stats.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    return str(preprocessor.corpus_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="위키피디아 전처리")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="데이터 디렉토리"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="최대 문서 수"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="API 사용 (기본: 덤프)"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="이미 다운로드된 데이터만 정제"
    )
    args = parser.parse_args()

    if args.clean_only:
        preprocessor = WikipediaPreprocessor(args.data_dir)

        if preprocessor.raw_path.exists():
            with open(preprocessor.raw_path, "r", encoding="utf-8") as f:
                articles = (json.loads(line) for line in f)
                clean_articles = preprocessor.process_articles(articles, save_raw=False)
                preprocessor.save_clean_articles(clean_articles)
                preprocessor.build_corpus()
        else:
            print(f"원본 데이터가 없습니다: {preprocessor.raw_path}")
    else:
        corpus_path = preprocess_wikipedia(
            data_dir=args.data_dir,
            max_articles=args.max_articles,
            use_api=args.use_api,
        )
        print(f"\n코퍼스 경로: {corpus_path}")


if __name__ == "__main__":
    main()
