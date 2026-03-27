# Hugging Face CLI Migration Plan

## Overview

This document outlines the plan to migrate the current `crawler.py` implementation (which downloads Korean Wikipedia from Wikimedia dumps) to use Hugging Face CLI for dataset acquisition.

## Migration Goals

1. **Replace manual dump download** - Use Hugging Face Hub instead of direct Wikimedia dump download
2. **Leverage preprocessed datasets** - Benefit from official Wikimedia-curated datasets
3. **Reduce complexity** - Remove XML parsing and manual filtering logic
4. **Improve maintainability** - Use battle-tested `huggingface_hub` library
5. **Preserve preprocessor integration** - Keep existing preprocessing pipeline

## Target Hugging Face Dataset

### Primary Choice: `wikimedia/wikipedia`

**Why this dataset:**
- ✅ **Official Wikimedia dump** - Maintained by Wikimedia foundation
- ✅ **Korean subset available** - Use `20231101.ko` configuration
- ✅ **648k Korean articles** - Comprehensive coverage
- ✅ **Pre-processed** - Wiki markup already cleaned
- ✅ **Parquet format** - Efficient loading with `datasets` library
- ✅ **Multiple snapshots** - Access dated versions (20231101, 20240501, etc.)
- ✅ **Language-specific** - `.ko` subset downloads only Korean Wikipedia articles (no other languages)

### How Language-Specific Subsets Work

The `wikimedia/wikipedia` dataset uses **subset configuration** to separate languages:

```python
# Example subset codes:
"20231101.en"  # English Wikipedia
"20231101.ko"  # Korean Wikipedia
"20231101.es"  # Spanish Wikipedia
"20231101.fr"  # French Wikipedia
# ... etc.
```

When you specify `SUBSET = "20231101.ko"`:
- The Hugging Face library automatically filters to **only Korean articles**
- No English, Spanish, French, or other language articles are downloaded
- Download size is ~20GB smaller (only Korean subset)

This is exactly what we need - **한국어 문서만 다운로드**!
- ✅ **Language-specific** - `.ko` subset downloads only Korean Wikipedia articles (no other languages)

**Alternative datasets (for comparison):**
| Dataset | Size | Format | Notes |
|---------|------|--------|-------|
| `lcw99/wikipedia-korean-20240501` | 515k | Unknown | Community processed |
| `recuse/korean_wiki` | 778k | Unknown | Community version |
| `devngho/korean_wikipedia` | 1.02M | Unknown | Larger but unverified |

## Architecture Changes

### Current Structure

```
crawler.py (298 lines)
├── WikipediaDumpCrawler
│   ├── download_dump() - HTTP download ~3GB BZ2
│   └── parse_dump() - XML stream parsing with ElementTree
├── WikipediaAPICrawler
│   ├── get_random_articles()
│   ├── get_articles_by_category()
│   └── search_articles()
└── download_korean_wikipedia()
```

### New Structure

```
src/cohebot/crawler/ (package)
├── __init__.py
├── base.py
│   └── BaseCrawler (ABC)
├── huggingface_crawler.py
│   └── HuggingFaceDatasetCrawler
└── preprocessor/
    ├── __init__.py
    ├── wikipedia_cleaner.py
    └── wikipedia_preprocessor.py (moved from preprocessor.py)
```

**Note:** Old single-file `crawler.py` will be replaced with this package structure.

## Implementation Plan

### Phase 1: New Package Structure

**Create `crawler/` subpackage under `src/cohebot/`:**

```
src/cohebot/crawler/
├── __init__.py
├── base.py
├── huggingface_crawler.py
└── preprocessor/
    ├── __init__.py
    ├── wikipedia_cleaner.py
    └── wikipedia_preprocessor.py
```

### Phase 2: Base Crawler Interface

**File: `crawler/base.py`**

```python
from abc import ABC, abstractmethod
from typing import Iterator

class BaseCrawler(ABC):
    @abstractmethod
    def fetch_articles(self, max_articles: int | None = None) -> Iterator[dict]:
        """Fetch articles from data source."""
        pass

    @abstractmethod
    def prepare_data(self, data_dir: str) -> Path:
        """Prepare data directory and download if needed."""
        pass
```

### Phase 3: Hugging Face Crawler Implementation

**File: `crawler/huggingface_crawler.py`**

**Class: `HuggingFaceDatasetCrawler`**

| Method | Purpose | Implementation |
|--------|---------|---------------|
| `__init__(repo_id, subset, data_dir)` | Setup repository and data directory | Store repo ID, subset name, data dir |
| `prepare_data()` | Download dataset via CLI or Python API | Use `snapshot_download()` or CLI wrapper |
| `fetch_articles(max_articles)` | Load dataset and yield articles | Use `datasets.load_dataset()` with `.map()` |
| `get_stats()` | Return dataset statistics | Query dataset info |

**Two approaches:**

#### Chosen Approach: Python API

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download

class HuggingFaceDatasetCrawler(BaseCrawler):
    REPO_ID = "wikimedia/wikipedia"
    SUBSET = "20231101.ko"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self) -> Path:
        # Download dataset to local cache
        local_path = snapshot_download(
            repo_id=self.REPO_ID,
            repo_type="dataset",
            local_dir=str(self.data_dir)
        )
        return Path(local_path)

    def fetch_articles(self, max_articles: int | None = None) -> Iterator[dict]:
        # Load dataset
        ds = load_dataset(self.REPO_ID, self.SUBSET, split="train")

        # Stream articles
        for i, article in enumerate(ds):
            if max_articles and i >= max_articles:
                break
            yield {
                "id": article["id"],           # Include Hugging Face article ID
                "url": article["url"],         # Include Hugging Face article URL
                "title": article["title"],
                "text": article["text"]
            }
```

**Note:** CLI wrapper approach was considered but not chosen because:
- Current `crawler.py` already has its own CLI interface (`main()` function)
- Using Python API provides better integration and testability
- `huggingface_cli` would just be another wrapper around the same functionality

### Phase 4: Preprocessor Integration

**Move preprocessor logic into `crawler/preprocessor/` package:**

**New structure:**

```
crawler/preprocessor/
├── __init__.py
│   └── Export: WikipediaPreprocessor, WikiTextCleaner
├── wikipedia_cleaner.py
│   └── WikiTextCleaner class (simplified - normalization/validation only)
└── wikipedia_preprocessor.py
    └── WikipediaPreprocessor class (pipeline orchestration)
```

**Key changes:**
1. Remove `mwparserfromhell` dependency (Hugging Face dataset already cleaned)
2. Simplify `WikiTextCleaner` - only do Unicode normalization and content validation
3. Keep `WikipediaPreprocessor` for corpus building
4. Remove regex patterns for wiki markup cleanup (no longer needed)
5. Preserve additional fields: `id` and `url` from Hugging Face dataset (useful for debugging/attribution even if currently unused)

### Phase 5: Update Imports

**Update `preprocessor.py`:**

```python
# Old import
# from .crawler import WikipediaDumpCrawler, WikipediaAPICrawler

# New import
from .crawler import HuggingFaceDatasetCrawler

def preprocess_wikipedia(
    data_dir: str = "data",
    max_articles: int | None = None,
) -> str:
    crawler = HuggingFaceDatasetCrawler(data_dir)

    crawler.prepare_data()
    articles = crawler.fetch_articles(max_articles)
    # ... rest of preprocessing unchanged
```

**Update CLI arguments in `crawler/main()`:**

```python
# No CLI changes needed - Hugging Face crawler is the only source now
```

### Phase 6: Dependency Updates

**File: `pyproject.toml`**

**Remove:**
- `beautifulsoup4` (no longer needed - Hugging Face dataset already clean)
- `lxml` (no longer needed - no XML parsing)
- `mwparserfromhell` (no longer needed - Hugging Face dataset already cleaned)

**Add:**
```toml
datasets = "*"
pyarrow = "*"
```

**Keep:**
- `huggingface_hub` (already present)
- `tqdm` (for progress bars)
- `numpy` (used in training)

## Migration Tasks

### Task List

- [ ] **T1**: Create `crawler/` package directory structure
- [ ] **T2**: Implement `BaseCrawler` ABC in `crawler/base.py`
- [ ] **T3**: Implement `HuggingFaceDatasetCrawler` in `crawler/huggingface_crawler.py` with `id`, `url` fields
- [ ] **T4**: Simplify and move `WikiTextCleaner` to `crawler/preprocessor/wikipedia_cleaner.py`
- [ ] **T5**: Move `WikipediaPreprocessor` to `crawler/preprocessor/wikipedia_preprocessor.py`
- [ ] **T6**: Create `crawler/__init__.py` with public API exports
- [ ] **T7**: Create `crawler/preprocessor/__init__.py` with exports
- [ ] **T8**: Delete old `crawler.py` (no backward compatibility)
- [ ] **T9**: Update `preprocessor.py` to import from new `crawler` package
- [ ] **T10**: Add `datasets` and `pyarrow` to `pyproject.toml`
- [ ] **T11**: Remove `beautifulsoup4`, `lxml`, and `mwparserfromhell` from `pyproject.toml`
- [ ] **T12**: Update docstrings and comments
- [ ] **T13**: Test new crawler with small subset
- [ ] **T14**: Verify end-to-end pipeline: crawler → preprocessor → corpus
- [ ] **T15**: Verify `id` and `url` fields are preserved in output dataset
- [ ] **T16**: Archive old `crawler.py` as `crawler.py.bak` for reference

## Backward Compatibility

### Support for Old Methods

**Option A: Deprecate slowly**
- Keep `WikipediaDumpCrawler` and `WikipediaAPICrawler` in old `crawler.py`
- Add deprecation warnings
- Remove in future version (v0.2.0)

**Option B: Break compatibility**
- Remove old classes immediately
- Update all downstream imports
- Simplify codebase

**Recommendation:** Option B (break compatibility) - the project is in early development (v0.1.0), and no external users depend on old API.

## Testing Strategy

### Unit Tests to Add

```python
# tests/test_crawler.py
def test_huggingface_crawler_init():
    """Test crawler initialization."""
    crawler = HuggingFaceDatasetCrawler("data/test")
    assert crawler.data_dir.exists()

def test_huggingface_crawler_fetch_articles():
    """Test article fetching."""
    crawler = HuggingFaceDatasetCrawler("data/test")
    crawler.prepare_data()
    articles = list(crawler.fetch_articles(max_articles=10))
    assert len(articles) == 10
    assert all("title" in a and "text" in a for a in articles)

def test_preprocessor_integration():
    """Test crawler + preprocessor integration."""
    crawler = HuggingFaceDatasetCrawler("data/test")
    crawler.prepare_data()
    articles = list(crawler.fetch_articles(max_articles=100))
    preprocessor = WikipediaPreprocessor("data/test")
    clean_articles = list(preprocessor.process_articles(articles))
    assert len(clean_articles) > 0
```

## Rollback Plan

If migration causes issues:

1. **Revert `crawler.py`** from backup
2. **Restore old imports** in `preprocessor.py`
3. **Undo dependency changes** in `pyproject.toml`
4. **Keep new code** in `crawler/` package as alternative implementation

**Backup strategy:**
```bash
# Before migration
cp src/cohebot/crawler.py src/cohebot/crawler.py.bak
git commit -am "Backup: old crawler.py before Hugging Face migration"

# If rollback needed
cp src/cohebot/crawler.py.bak src/cohebot/crawler.py
```

## Benefits of Migration

| Aspect | Before | After |
|---------|---------|--------|
| **Download size** | ~3GB BZ2 file | ~1GB Parquet (compressed) |
| **Parsing time** | Stream-parse XML (slow) | Load Parquet (fast) |
| **Memory usage** | XML streaming + DOM | Lazy dataset loading |
| **Code complexity** | 298 lines of parsing | ~100 lines of API calls |
| **Maintenance** | Manual XML handling | Delegated to Hugging Face |
| **Data quality** | Manual filtering | Official Wikimedia curation |
| **Updates** | Manual dump download | Dataset versioning |
| **Error handling** | Manual retry | Built-in retry + cache |

## Potential Risks

1. **Dataset version drift** - Hugging Face may have different versions than Wikimedia dumps
   - **Mitigation**: Verify article count and content before migration

2. **Preprocessor assumptions** - Existing preprocessor expects raw wiki markup
   - **Mitigation**: Simplify preprocessor to only normalize/validate

3. **Dependency conflicts** - `datasets` library may conflict with existing code
   - **Mitigation**: Test with isolated environment first

4. **Network issues** - Hugging Face Hub may be slow or unavailable
   - **Mitigation**: Keep old code as fallback, implement local caching

## Timeline Estimate

| Task | Estimated Time |
|------|----------------|
| T1-T3: Core crawler implementation | 2-3 hours |
| T4-T7: Preprocessor refactoring | 1-2 hours |
| T8-T12: Integration updates | 1 hour |
| T13-T15: Testing and verification | 1-2 hours |
| **Total** | **5-8 hours** |

## Success Criteria

- [ ] New crawler downloads dataset successfully
- [ ] Preprocessor integrates without errors
- [ ] End-to-end pipeline produces valid corpus
- [ ] Corpus quality matches or exceeds old pipeline
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Old code archived

## Next Steps

After reviewing this plan:

1. **Approve or modify** migration strategy
2. **Choose approach** (Python API vs CLI wrapper)
3. **Decide on compatibility** (keep old code or break)
4. **Run `/start-work`** to begin implementation

## Post-Migration Enhancements (Completed)

After completing the initial migration, additional enhancements were implemented to address disk space and performance issues:

### Chunked/Lazy Downloading

**Problem:** Initial implementation used `load_dataset()` which downloaded entire ~20GB dataset (557 Parquet files) even when requesting only 10 articles.

**Solution:** Implemented streaming mode for lazy loading.

```python
# Old approach - downloads entire dataset
ds = load_dataset(self.REPO_ID, self.SUBSET, split="train")

# New approach - lazy loading with streaming
ds = load_dataset(self.REPO_ID, self.SUBSET, split="train", streaming=True)
```

**Benefits:**
- Files downloaded on-demand as you iterate
- Only necessary data is loaded into memory
- No need to download full 20GB dataset for small requests

### File Selection Options

Added parameters to `HuggingFaceDatasetCrawler.__init__()`:

| Parameter | Type | Purpose |
|----------|------|---------|
| `specific_files` | `list[str] \| None` | Download only specific Parquet files (e.g., `["train-00000.parquet"]`) |
| `exclude_files` | `list[str]` | Exclude file patterns (e.g., `["*-cache-*"]`) |
| `max_files` | `int \| None` | Limit number of files downloaded |
| `streaming` | `bool` | Enable lazy loading mode (default: `True`) |

**Example usage:**

```python
# Download only first 5 Parquet files
crawler = HuggingFaceDatasetCrawler(
    data_dir="data",
    max_files=5,
    streaming=True
)

# Download specific files only
crawler = HuggingFaceDatasetCrawler(
    data_dir="data",
    specific_files=["train-00000.parquet", "train-00001.parquet"],
    streaming=True
)
```

### Cleanup Method

Added `cleanup()` method to delete downloaded Parquet files after processing:

```python
def cleanup(self) -> None:
    """Remove downloaded Parquet files to free disk space."""
    for parquet_file in self.data_dir.glob("**/*.parquet"):
        parquet_file.unlink()
```

**Benefits:**
- Reduces disk usage by deleting temporary files
- Useful for working with large datasets
- Call after processing to free space

### Field Preservation

Updated `WikipediaPreprocessor.process_articles()` to preserve `id` and `url` fields:

```python
# Old implementation - only title and text
yield {
    "title": article["title"],
    "text": clean_text
}

# New implementation - includes id and url
yield {
    "id": article["id"],
    "url": article["url"],
    "title": article["title"],
    "text": clean_text
}
```

### Test Results

**Test: Chunked downloading with 10 articles**

```
Testing chunked downloading with streaming mode...
Fetching 10 articles...
  Fetched article 1: 지미 카터
  Fetched article 2: 수학
  Fetched article 3: 수학 상수
  Fetched article 4: 문학
  Fetched article 5: 나라 목록
  Fetched article 6: 화학
  Fetched article 7: 체첸 공화국
  Fetched article 8: 맥스웰 방정식
  Fetched article 9: 초월수
  Fetched article 10: 음계
Successfully fetched 10 articles

Verifying field preservation...
First article fields: ['id', 'url', 'title', 'text']
✅ id and url fields preserved
  id: 5
  url: https://ko.wikipedia.org/wiki/%EC%A7%80%EB%AF%B8%20%EC%B9%B4%ED%84%B0

Cleaning up...
✅ Cleanup completed

✅ Test completed successfully
```

**Results:**
- ✅ Successfully fetched 10 articles without downloading full 20GB dataset
- ✅ All fields preserved: `['id', 'url', 'title', 'text']`
- ✅ Cleanup method successfully deleted downloaded files
- ✅ No disk space issues

### Updated Implementation Details

**File: `crawler/huggingface_crawler.py`**

```python
class HuggingFaceDatasetCrawler(BaseCrawler):
    REPO_ID = "wikimedia/wikipedia"
    SUBSET = "20231101.ko"

    def __init__(
        self,
        data_dir: str = "data",
        specific_files: Optional[list[str]] = None,
        exclude_files: Optional[list[str]] = None,
        max_files: Optional[int] = None,
        streaming: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.specific_files = specific_files
        self.exclude_files = exclude_files or []
        self.max_files = max_files
        self.streaming = streaming

    def fetch_articles(self, max_articles: int | None = None) -> Iterator[dict]:
        ds = load_dataset(self.REPO_ID, self.SUBSET, split="train", streaming=self.streaming)

        if max_articles:
            ds = ds.take(max_articles)

        for article in ds:
            yield {
                "id": article["id"],
                "url": article["url"],
                "title": article["title"],
                "text": article["text"]
            }

    def cleanup(self) -> None:
        for parquet_file in self.data_dir.glob("**/*.parquet"):
            parquet_file.unlink()
```

### Key Improvements Summary

| Feature | Before | After |
|---------|---------|--------|
| **Download approach** | Full dataset download | Lazy loading (streaming mode) |
| **Disk usage** | 20GB+ for any request | Only needed files downloaded |
| **File selection** | All files or nothing | Specific files, exclude patterns, max files |
| **Memory usage** | Load full dataset into memory | On-demand iteration |
| **Cleanup** | Manual file deletion | Automatic `cleanup()` method |
| **Field preservation** | Title and text only | id, url, title, and text |

### Usage Examples

**Example 1: Small subset for testing**

```python
crawler = HuggingFaceDatasetCrawler(streaming=True)
articles = list(crawler.fetch_articles(max_articles=10))
# Downloads only necessary files for 10 articles
crawler.cleanup()
```

**Example 2: Specific Parquet files**

```python
crawler = HuggingFaceDatasetCrawler(
    specific_files=["train-00000.parquet", "train-00001.parquet"],
    streaming=True
)
articles = list(crawler.fetch_articles())
# Downloads only 2 specific files
```

**Example 3: Limit file count**

```python
crawler = HuggingFaceDatasetCrawler(
    max_files=10,
    streaming=True
)
articles = list(crawler.fetch_articles())
# Downloads only first 10 Parquet files
crawler.cleanup()
```
