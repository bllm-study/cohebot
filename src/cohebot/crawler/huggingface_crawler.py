from collections.abc import Iterator
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from .base import BaseCrawler


class HuggingFaceDatasetCrawler(BaseCrawler):
    """Crawler for Hugging Face datasets.

    Uses the official `wikimedia/wikipedia` dataset with Korean subset
    to fetch pre-processed Wikipedia articles.

    Supports lazy loading (streaming mode) and selective file downloading
    to reduce disk usage and improve performance.
    """

    REPO_ID = "wikimedia/wikipedia"
    SUBSET = "20231101.ko"

    def __init__(
        self,
        data_dir: str = "data",
        specific_files: list[str] | None = None,
        exclude_files: list[str] | None = None,
        max_files: int | None = None,
        streaming: bool = True,
    ):
        """Initialize crawler.

        Args:
            data_dir: Directory to store downloaded data.
            specific_files: List of specific Parquet files to download (e.g., ["train-00000.parquet"]).
                If None, all files are included.
            exclude_files: List of file patterns to exclude (e.g., ["*-cache-*"]).
            max_files: Maximum number of files to download. If None, all matching files.
            streaming: Use streaming mode (lazy loading) instead of loading entire dataset.
                Streaming mode downloads files on-demand as you iterate.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.specific_files = specific_files
        self.exclude_files = exclude_files or []
        self.max_files = max_files
        self.streaming = streaming

    def prepare_data(self, data_dir: str | None = None) -> Path:
        """Download dataset from Hugging Face Hub.

        Uses huggingface_hub's snapshot_download() to cache and
        download the dataset to local directory.

        Supports selective file downloading based on init parameters:
        - specific_files: Only download listed files
        - exclude_files: Exclude matching file patterns
        - max_files: Limit number of files downloaded

        Args:
            data_dir: Data directory (overrides self.data_dir if provided).

        Returns:
            Path to downloaded dataset directory.
        """
        target_dir = Path(data_dir) if data_dir else self.data_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        allow_patterns = self.specific_files
        ignore_patterns = self.exclude_files

        local_path = snapshot_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        if self.max_files:
            downloaded_files = list(Path(local_path).glob("*.parquet"))
            files_to_keep = sorted(downloaded_files, key=lambda p: p.name)[: self.max_files]
            for file_to_delete in set(downloaded_files) - set(files_to_keep):
                file_to_delete.unlink()

        return Path(local_path)

    def fetch_articles(self, max_articles: int | None = None) -> Iterator[dict]:
        """Fetch articles from Hugging Face dataset.

        Loads the dataset using the `datasets` library with streaming mode.
        Downloads files on-demand as you iterate through articles.
        Includes id, url, title, and text fields.

        Args:
            max_articles: Maximum number of articles to fetch. None for all.

        Yields:
            Article dictionaries with id, url, title, and text fields.
        """
        ds = load_dataset(
            self.REPO_ID,
            self.SUBSET,
            split="train",
            streaming=self.streaming,
            cache_dir=str(self.data_dir),
        )

        if max_articles:
            ds = ds.take(max_articles)

        for article in ds:
            yield {
                "id": article["id"],
                "url": article["url"],
                "title": article["title"],
                "text": article["text"],
            }

    def cleanup(self) -> None:
        """Remove downloaded Parquet files to free disk space.

        Deletes all .parquet files in the data directory after processing.
        Use this when working with large datasets to manage disk usage.
        """
        for parquet_file in self.data_dir.glob("**/*.parquet"):
            parquet_file.unlink()
