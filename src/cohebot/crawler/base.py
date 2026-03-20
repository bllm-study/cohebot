from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator


class BaseCrawler(ABC):
    """Base abstract class for data crawlers.

    All crawler implementations must inherit from this class
    and implement the abstract methods.
    """

    @abstractmethod
    def fetch_articles(self, max_articles: int | None = None) -> Iterator[dict]:
        """Fetch articles from data source.

        Args:
            max_articles: Maximum number of articles to fetch. None for all.

        Yields:
            Article dictionaries with at minimum 'title' and 'text' fields.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

    @abstractmethod
    def prepare_data(self, data_dir: str) -> Path:
        """Prepare data directory and download if needed.

        Args:
            data_dir: Path to data directory.

        Returns:
            Path to prepared data directory.

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
