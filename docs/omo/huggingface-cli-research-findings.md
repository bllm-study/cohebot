# Hugging Face CLI Research Findings

## Research Summary

This document captures all research findings conducted to support the migration from custom Wikipedia crawler to Hugging Face CLI-based data acquisition.

## Research Questions & Answers

### Q1: How to use huggingface-cli for dataset downloads?

**Findings:**

The CLI has been renamed from `huggingface-cli` to `hf` (new syntax).

**Core commands:**
| Legacy | New `hf` | Purpose |
|--------|----------|---------|
| `huggingface-cli login` | `hf auth login` | Login to HuggingFace |
| `huggingface-cli download` | `hf download` | Download files |

**Basic download syntax:**
```bash
# Download entire dataset
hf download wikimedia/wikipedia --repo-type dataset --local-dir ./ko-wiki

# With specific revision
hf download wikimedia/wikipedia --repo-type dataset --revision 20231101.ko --local-dir ./ko-wiki

# With cache directory
hf download --repo-type dataset wikimedia/wikipedia --cache-dir /custom/cache
```

**Key options:**
- `--repo-type {dataset,model,space}` - Type of repository
- `--local-dir` - Download to specific directory (copies from cache)
- `--cache-dir` - Where to store cached files
- `--revision` - Specific git commit/branch/tag
- `--token` - HuggingFace API token
- `--max-workers` - Parallel download threads
- `--include` / `--exclude` - Filter files by pattern
- `--force` - Force re-download (ignore cache)

**Source:** Official Hugging Face documentation and CLI help output.

---

### Q2: What Korean Wikipedia datasets are available on Hugging Face?

**Findings:**

| Dataset | Rows | Format | Notes |
|---------|------|--------|-------|
| **`wikimedia/wikipedia`** ⭐ | 648k (ko) | Parquet | **Official Wikimedia dump** - use `20231101.ko` subset |
| `lcw99/wikipedia-korean-20240501` | 515k | - | Pre-processed Korean wiki |
| `lcw99/wikipedia-korean-20221001` | 607k | - | Older dump version |
| `recuse/korean_wiki` | 778k | - | Community processed |
| `devngho/korean_wikipedia` | 1.02M | - | Large community version |
| `shopkeeper/korean_wiki_content_only_120125` | 4.17M | - | Content only (no metadata) |

**Recommended:** `wikimedia/wikipedia` with `20231101.ko` configuration

**Reasons for recommendation:**
- ✅ Official Wikimedia dump - Maintained by Wikimedia foundation
- ✅ Cleaned text (markdown stripped)
- ✅ 648k Korean articles - Comprehensive coverage
- ✅ Parquet format - Efficient loading with `datasets` library
- ✅ Multiple snapshots - Access dated versions (20231101, 20240501, etc.)

**Source:** Hugging Face Hub dataset search and dataset cards.

---

### Q3: How does `wikimedia/wikipedia` dataset compare to current crawler output?

**Findings:**

| Aspect | Current Crawler | Hugging Face Dataset |
|--------|---------------------|----------------------|
| **Data Source** | Wikimedia dumps server (manual download) | Hugging Face Hub (official) |
| **Redirect filtering** | ✅ Excludes (`#redirect`, `#넘겨주기`) | ✅ Excluded (`ns != "0"` or `redirect is not None`) |
| **Namespace filtering** | ✅ Excludes non-main NS | ✅ Only namespace 0 included |
| **Text cleaning** | ✅ Removes wiki markup (regex/mwparserfromhell) | ✅ Uses `mwparserfromhell.strip_code()` |
| **Output fields** | `{"title": str, "text": str}` | `{"id": str, "url": str, "title": str, "text": str}` |
| **Language aliases** | Korean-specific prefixes | Korean aliases included |
| **Data format** | XML → JSONL (manual parsing) | Parquet (columnar storage) |
| **File size** | ~3GB BZ2 (before parsing) | 783MB Parquet (after processing) |
| **Parsing complexity** | 298 lines of XML parsing code | ~50 lines of API calls |

**Compatibility analysis:**
- ✅ **Field compatible**: Both have `title` and `text` fields
- ✅ **Content compatible**: Both produce clean plain text (no wiki markup)
- ✅ **Filtering compatible**: Both exclude redirects and special namespaces
- ℹ️  **Additional fields**: Hugging Face has `id` and `url` (can be preserved in output)

**Source:** Direct dataset inspection and wikimedia/wikipedia dataset card.

---

### Q4: What preprocessing is already applied to Hugging Face dataset?

**Findings:**

**Wiki markup is fully removed** using `mwparserfromhell` library.

From dataset processing code ([wikipedia.py lines 1216-1272](https://huggingface.co/datasets/wikimedia/wikipedia/blob/script/wikipedia.py)):

```python
def _parse_and_clean_wikicode(raw_content, parser, language):
    """Strip formatting and unwanted sections from raw page content."""
    wikicode = parser.parse(raw_content)

    # 1. Filters for magic words (parser instructions like __NOTOC__)
    re_rm_magic = re.compile("__[A-Z]*__", flags=re.UNICODE)

    # 2. Filters for file/image/media links
    media_prefixes = "|".join(["File", "Image", "Media"] + MEDIA_ALIASES.get(language, []))
    re_rm_wikilink = re.compile(f"^(?:{media_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    # 3. Strips category prefixes (e.g., "Category:Towns" → "Towns")
    cat_prefixes = "|".join(["Category"] + CAT_ALIASES.get(language, []))
    re_clean_wikilink = re.compile(f"^(?:{cat_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    # 4. Uses mwparserfromhell's strip_code() to convert wikitext → plain text
    section_text.append(re.sub(re_rm_magic, "", section.strip_code().strip()))
```

**Implications:**
- ✅ No need for wiki markup removal regex in new preprocessor
- ✅ No need for redirect detection (already excluded)
- ✅ No need for namespace filtering (already excluded)
- ⚠️  `mwparserfromhell` dependency can be removed from project

**Source:** Dataset processing script source code.

---

### Q5: What are best practices for organizing Python modules with crawlers and preprocessors?

**Findings:**

#### Recommended Pattern: Subpackage with Facade

From analysis of production projects (scrapy, feapder, crawlee-python, CodeFuse-muAgent, voilib):

```
mypackage/
├── __init__.py           # Public API facade
├── crawler/              # Crawler subpackage
│   ├── __init__.py       # Crawler public API
│   ├── base.py           # Base crawler class (ABC)
│   ├── http_crawler.py   # HTTP implementation
│   └── strategies/       # Crawling strategies
├── processor/            # Preprocessor subpackage
│   ├── __init__.py       # Processor public API
│   ├── base.py           # Base processor class
│   └── text_processor.py # Text preprocessing
├── pipeline/             # Pipeline orchestration
│   └── crawler_pipeline.py
└── models/               # Data models/schemas
    └── schemas.py        # Pydantic models
```

#### Key Principles:

1. **Package Layout** - Use subpackages for separation of concerns
2. **Public API** - Use `__init__.py` as facade to expose stable interfaces
3. **Abstract Base Classes** - Define `BaseCrawler(ABC)` for extensibility
4. **Type Safety** - Use dataclasses/Pydantic for input/output schemas
5. **Pipeline Composition** - Prefer composition over inheritance

**Source:** GitHub analysis of 5+ production scraping/crawling projects.

---

## Why This Research Was Conducted

### Objective
To evaluate whether migrating from custom Wikipedia crawler to Hugging Face CLI-based data acquisition is feasible and beneficial.

### Decision Factors Considered

| Factor | Question | Answer | Impact |
|--------|-----------|---------|--------|
| **Data Compatibility** | Can Hugging Face dataset produce same corpus? | ✅ Yes - fields and content match | High |
| **Preprocessing Overlap** | Is current preprocessing redundant? | ✅ Yes - Hugging Face already cleans | High |
| **Code Complexity** | Will migration simplify codebase? | ✅ Yes - 298 lines → ~50 lines | High |
| **Maintenance** | Is Hugging Face better maintained? | ✅ Yes - official dataset | Medium |
| **Performance** | Will it be faster/smaller? | ✅ Yes - 783MB vs 3GB | Medium |
| **Dependencies** | Will we reduce dependencies? | ✅ Yes - remove 3 packages | Low |

### Conclusion

**Migration is highly recommended** because:
1. Data compatibility is confirmed (same fields and content)
2. Code complexity reduces significantly
3. Maintenance burden shifts to official maintainers
4. Preprocessing logic can be simplified
5. Better performance (smaller downloads, faster loading)

---

## Key Recommendations

### 1. Use Python API (Not CLI wrapper)

**Rationale:**
- Current `crawler.py` already has its own CLI interface
- Python API provides better testability
- CLI would be just another wrapper

**Implementation:**
```python
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
for article in ds:
    yield {"title": article["title"], "text": article["text"]}
```

### 2. Preserve Additional Fields (id, url)

**Rationale:**
- Hugging Face dataset includes `id` and `url` fields
- These are useful for debugging, tracking, and attribution
- Even if unused now, they add value to dataset

**Implementation:**
```python
# Save all fields from Hugging Face dataset
{
    "id": article["id"],
    "url": article["url"],
    "title": article["title"],
    "text": article["text"]
}
```

### 3. Simplify Preprocessor

**Rationale:**
- Hugging Face dataset already has wiki markup removed
- Redirects and special namespaces already excluded
- Preprocessor only needs to do: Unicode normalization, content validation, length limits

**Remove from preprocessor:**
- ❌ Wiki markup regex patterns
- ❌ Redirect detection logic
- ❌ Namespace filtering logic
- ❌ `mwparserfromhell` dependency

**Keep in preprocessor:**
- ✅ Unicode NFC normalization
- ✅ Korean content ratio validation (≥30%)
- ✅ Text length limits
- ✅ Corpus building with `

` separators

### 4. Break Backward Compatibility

**Rationale:**
- Only `preprocessor.py` directly imports from `crawler.py`
- `train.py` uses `preprocessor` indirectly
- No external API users (private project, v0.1.0)
- Early development stage - no breaking change impact

**Action:**
- Remove old `WikipediaDumpCrawler` and `WikipediaAPICrawler`
- No deprecation period needed

---

## References

### Primary Sources
1. **Hugging Face Hub** - Dataset cards and documentation
   - `wikimedia/wikipedia`: https://huggingface.co/datasets/wikimedia/wikipedia
   - Dataset processing: https://huggingface.co/datasets/wikimedia/wikipedia/blob/script/wikipedia.py

2. **Hugging Face CLI Documentation**
   - Official docs: https://huggingface.co/docs/huggingface_hub/guides/cli
   - CLI commands: `hf download --help`

3. **Production Projects Analysis**
   - scrapy: https://github.com/scrapy/scrapy
   - crawlee-python: https://github.com/crawlee/python
   - feapder: https://github.com/Boris-code/feapder
   - CodeFuse-muAgent: https://github.com/codefuse-ai/CodeFuse-muAgent
   - voilib: https://github.com/unmonoqueteclea/voilib

### Data Sources Consulted
- Official Wikimedia dumps: https://dumps.wikimedia.org/kowiki/latest/
- Korean Wikipedia API: https://ko.wikipedia.org/w/api.php
- Hugging Face dataset search and comparison

---

## Timeline

| Date | Research Task | Duration |
|-------|--------------|----------|
| 2026-03-21 | Hugging Face CLI documentation review | ~15 min |
| 2026-03-21 | Korean Wikipedia dataset search | ~20 min |
| 2026-03-21 | Dataset compatibility analysis | ~25 min |
| 2026-03-21 | Module organization patterns research | ~30 min |
| 2026-03-21 | Documentation compilation | ~15 min |
| **Total** | | **~1.75 hours** |

---

## Next Steps

See [`huggingface-cli-migration-plan.md`](./huggingface-cli-migration-plan.md) for implementation plan based on these findings.
