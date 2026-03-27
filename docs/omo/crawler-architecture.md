# Current Crawler Architecture Documentation

## Overview

This document describes the current implementation of the `crawler.py` module in the cohebot project, which handles Korean Wikipedia data collection.

## Current Project Structure

```
src/cohebot/
├── __init__.py           # Package marker
├── attention/            # Attention mechanisms (flash, gqa, mha, rope)
├── crawler.py           # ⭐ MAIN SUBJECT - Wikipedia data collection
├── dataset.py           # PyTorch Dataset/DataLoader
├── generate.py          # Text generation
├── model.py            # GPT-2 model implementation
├── preprocessor.py     # Wiki markup cleaning & corpus building
├── tokenizer.py        # GPT2 tokenizer wrapper
└── train.py            # Training loop & checkpointing
```

## Crawler Module (`crawler.py`)

### Classes

#### 1. `WikipediaDumpCrawler`

Downloads and parses Korean Wikipedia dump files from Wikimedia.

**Responsibilities:**
- Download BZ2-compressed XML dump from Wikimedia dumps server
- Stream-parse XML with `xml.etree.ElementTree` (stdlib)
- Filter out redirect pages
- Filter out special namespace pages (files, templates, categories, etc.)
- Yield articles as `{"title": str, "text": str}`

**Key Methods:**

| Method | Purpose | Returns |
|--------|---------|---------|
| `__init__(data_dir)` | Setup data directory and dump path | `None` |
| `download_dump(force=False)` | Download kowiki-latest-pages-articles.xml.bz2 | `Path` |
| `parse_dump(max_articles=None)` | Stream-parse dump file and yield articles | `Iterator[dict]` |

**Constants:**
- `DUMP_BASE_URL = "https://dumps.wikimedia.org/kowiki/latest/"`
- `DUMP_FILENAME = "kowiki-latest-pages-articles.xml.bz2"`

**Filtering Logic:**
1. Skip pages with text starting with `#redirect` or `#넘겨주기`
2. Skip pages with `:` in title (special namespaces)
3. Exclude prefixes: 위키백과, Wikipedia, 파일, File, 틀, Template, 분류, Category, 포털, Portal, 모듈, Module, 사용자, User, 토론, Talk, 미디어위키, MediaWiki, 도움말, Help

#### 2. `WikipediaAPICrawler`

Fetches articles via Wikipedia MediaWiki API.

**Responsibilities:**
- Query Wikipedia API for random articles
- Query by category for topic-specific collection
- Search articles by query string
- Handle API pagination
- Yield articles as `{"title": str, "text": str}`

**Key Methods:**

| Method | Purpose | Parameters | Returns |
|--------|---------|-------------|---------|
| `get_random_articles(count=100)` | Fetch random articles | `count` | `Iterator[dict]` |
| `get_articles_by_category(category, max_articles=1000)` | Fetch articles by category | `category`, `max_articles` | `Iterator[dict]` |
| `search_articles(query, max_articles=100)` | Search and fetch articles | `query`, `max_articles` | `Iterator[dict]` |

**Constants:**
- `API_URL = "https://ko.wikipedia.org/w/api.php"`
- Uses User-Agent: "GPT2-Study-Bot/1.0 (Educational Purpose)"

### Helper Functions

#### `download_korean_wikipedia(data_dir, max_articles)`

Convenience function combining dump download and parsing.

**Flow:**
1. Create `WikipediaDumpCrawler`
2. Call `download_dump()`
3. Call `parse_dump(max_articles)`
4. Return list of articles

### CLI Entry Point

#### `main()`

Command-line interface for crawling.

**Arguments:**
- `--method`: `dump` or `api` (default: `dump`)
- `--max-articles`: Limit number of articles (optional)
- `--data-dir`: Data storage directory (default: `data`)

**Usage:**
```bash
# Using dump (default)
python -m cohebot.crawler --method dump --max-articles 1000

# Using API
python -m cohebot.crawler --method api --max-articles 100
```

## Integration with Other Modules

### Dependency: `preprocessor.py`

`preprocessor.py` imports from `crawler.py`:

```python
from .crawler import WikipediaDumpCrawler, WikipediaAPICrawler
```

**Usage in `preprocess_wikipedia()`:**

```python
if use_api:
    crawler = WikipediaAPICrawler(data_dir)
    articles = crawler.get_random_articles(max_articles or 1000)
else:
    crawler = WikipediaDumpCrawler(data_dir)
    crawler.download_dump()
    articles = crawler.parse_dump(max_articles)
```

### Downstream Usage

1. **`preprocessor.py`**: Directly imports and uses crawler classes
2. **`train.py`**: Uses `preprocess_wikipedia()` which internally uses crawler
3. **CLI**: `cohebot-crawl` entry point in `pyproject.toml`

## Data Flow

```
┌─────────────────────────┐
│ Wikipedia Dump Server   │
│ (kowiki-latest-       │
│  pages-articles.xml.bz2)│
└──────────┬────────────┘
           │ HTTP + stream
           ▼
┌─────────────────────────────────┐
│ WikipediaDumpCrawler          │
│ - download_dump()            │
│ - parse_dump()               │
└──────────┬──────────────────┘
           │ yields {"title", "text"}
           ▼
┌─────────────────────────────────┐
│ WikipediaPreprocessor         │
│ (preprocessor.py)            │
│ - process_articles()          │
│ - save_clean_articles()       │
│ - build_corpus()             │
└──────────┬──────────────────┘
           │
           ├─────────────────┬─────────────────┐
           ▼                 ▼                 ▼
    wiki_raw.jsonl    wiki_clean.jsonl   wiki_corpus.txt
```

## Current Dependencies

### Direct Dependencies (crawler.py)

| Package | Purpose | Usage |
|---------|---------|-------|
| `bz2` (stdlib) | Decompress BZ2 files | `bz2.open()` |
| `xml.etree.ElementTree` (stdlib) | Parse XML dumps | `ET.iterparse()` |
| `requests` | HTTP downloads | `requests.get()` |
| `tqdm` | Progress bars | `tqdm()` |

### Declared Dependencies (pyproject.toml)

| Package | Version | Used in crawler? |
|---------|---------|-----------------|
| `requests>=2.31.0` | ✅ Yes | HTTP downloads |
| `tqdm>=4.65.0` | ✅ Yes | Progress bars |
| `beautifulsoup4>=4.12.0` | ❌ No | Declared but unused |
| `lxml>=5.0.0` | ❌ No | Declared but unused |
| `mwparserfromhell>=0.6.5` | ⚠️ Optional | Used in preprocessor only |
| `huggingface_hub>=1.7.2` | ❌ No | Currently unused |

## Current Limitations

### 1. Manual XML Parsing
- Uses `xml.etree.ElementTree` with manual event-driven parsing
- Custom filtering logic for redirects and special namespaces
- Vulnerable to XML structure changes

### 2. Large File Downloads
- Downloads entire ~3GB BZ2 dump file
- No incremental or differential updates
- Requires significant disk space

### 3. Limited API Methods
- Only three API methods: random, category, search
- No pagination optimization
- No caching mechanism

### 4. Hardcoded Korean-Specific Logic
- Korean Wikipedia URL hardcoded
- Korean namespace filtering hardcoded
- Korean redirect markers (`#넘겨주기`) hardcoded

### 5. No Error Recovery
- No retry logic for failed downloads
- No checkpointing for partial parsing
- No validation of downloaded data integrity

## Data Storage

### Files Generated

| File | Format | Purpose |
|------|---------|---------|
| `data/kowiki-latest-pages-articles.xml.bz2` | BZ2-compressed XML | Raw dump (from crawler) |
| `data/wiki_raw.jsonl` | JSONL | Raw article collection (optional) |
| `data/wiki_clean.jsonl` | JSONL | Cleaned articles (from preprocessor) |
| `data/wiki_corpus.txt` | Plain text | Final training corpus (from preprocessor) |

## Summary

The current `crawler.py` implements two approaches for collecting Korean Wikipedia data:

1. **Dump Download** (default) - Downloads full XML dump and stream-parses
2. **API Crawling** - Fetches articles via MediaWiki API

## Language-Specific Data Acquisition

### Hugging Face Dataset (New Implementation)

The `HuggingFaceDatasetCrawler` uses **language-specific subset** from `wikimedia/wikipedia`:

```python
REPO_ID = "wikimedia/wikipedia"
SUBSET = "20231101.ko"  # ← .ko = 한국어 (Korean)
```

**What this means:**
- **한국어 문서만 다운로드됨** - The `.ko` subset contains only Korean Wikipedia articles
- **이미 정제됨** - Hugging Face가 wiki markup을 이미 제거
- **리다이렉트 및 특수 네임스페이스 자동 필터링됨** - Redirects 및 특수 네임스페이스가 자동으로 제외됨
- **공식 데이터셋** - 위키미디어 재단에서 유지보관되는 648k 한국어 문서

### 언어별 섭브셋 안내

The `wikimedia/wikipedia` dataset supports different language subsets:

| 언어 | 섭브셋 코드 | 문서 수 |
|------|----------|--------|
| 영어 | `20231101.en` | - |
| 한국어 | `20231101.ko` ⭐ | 648k |
| 스페인어 | `20231101.es` | - |
| 프랑스어 | `20231101.fr` | - |
| 독일어 | `20231101.de` | - |
| 이탈리아어 | `20231101.it` | - |

현재 코드는 `.ko` 섭브셋을 사용하므로 **한국어 문서만 다운로드**합니다.

### Comparison

| 항목 | 옛은 방식 | 새 방식 (Hugging Face) |
|------|-----------|----------------|
| **데이터 출처** | Wikimedia 덤프 서버 | Hugging Face Hub |
| **파싱 포맷** | XML (수동 파싱) | Parquet (컬럼너) |
| **파싱 크기** | ~3GB BZ2 + XML 스트리밍 | ~20GB Parquet (전체) |
| **파싱 속도** | 느림 | 빠름 |
| **전처리 필요** | ✅ (직접 정규식/필터링) | ❌ (이미 정제됨) |
| **코드 복잡성** | 298줄 (XML 파싱) | ~70줄 (API 호출) |
| **다운로드 대상** | 전체 덤프 (기본) | 언어별 섭브셋 (.ko) |
| **데이터 품질** | 수동 정제 후 품질 동일 | 공식 정제 품질 보증 |
| **버전 관리** | 수동 덤프 재다운로드 | 데이터셋 버전 선택 |

## Next Steps

See [`huggingface-cli-migration-plan.md`](./huggingface-cli-migration-plan.md) for the planned migration to Hugging Face CLI-based data acquisition.
