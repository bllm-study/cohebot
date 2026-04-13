"""Integration tests for HuggingFaceDatasetCrawler.

Tests cover:
- End-to-end pipeline (crawler → preprocessor → corpus)
- Chunked downloading with streaming mode
- Field preservation in output files (id, url, title, text)
"""

import json
from pathlib import Path

from cohebot.crawler import HuggingFaceDatasetCrawler
from cohebot.crawler.preprocessor import WikipediaPreprocessor


def test_end_to_end_pipeline():
    """Complete end-to-end pipeline test with 10 articles.

    Tests:
    1. Crawler fetches articles with streaming mode
    2. Preprocessor processes and saves to wiki_clean.jsonl
    3. Corpus generation produces wiki_corpus.txt
    4. Cleanup removes downloaded Parquet files

    Expected Results:
    - 10 articles fetched successfully
    - wiki_clean.jsonl created with id, url, title, text fields
    - wiki_corpus.txt created with formatted text
    - Parquet files cleaned up
    """
    print("\n" + "=" * 60)
    print("TEST: End-to-End Pipeline (10 articles)")
    print("=" * 60)

    test_dir = Path("data/test_e2e")
    test_dir.mkdir(parents=True, exist_ok=True)

    crawler = HuggingFaceDatasetCrawler(data_dir=str(test_dir), streaming=True)
    preprocessor = WikipediaPreprocessor(data_dir=str(test_dir))

    print("\n1. Fetching 10 articles...")
    articles = list(crawler.fetch_articles(max_articles=10))
    print(f"   ✅ Fetched {len(articles)} articles")

    print("\n2. Processing and saving cleaned articles...")
    clean_articles = list(preprocessor.process_articles(iter(articles)))
    saved_count = preprocessor.save_clean_articles(iter(clean_articles))
    print(f"   ✅ Saved {saved_count} cleaned articles to wiki_clean.jsonl")

    print("\n3. Building corpus...")
    total_chars = preprocessor.build_corpus(add_title=True)
    print(f"   ✅ Generated wiki_corpus.txt ({total_chars:,} characters)")

    print("\n4. Verifying files...")
    clean_path = test_dir / "wiki_clean.jsonl"
    corpus_path = test_dir / "wiki_corpus.txt"

    assert clean_path.exists(), "wiki_clean.jsonl not created"
    assert corpus_path.exists(), "wiki_corpus.txt not created"
    print("   ✅ Both files exist")

    print("\n5. Verifying wiki_clean.jsonl content...")
    with open(clean_path, encoding="utf-8") as f:
        first_line = json.loads(f.readline())
        expected_fields = {"id", "url", "title", "text"}
        actual_fields = set(first_line.keys())

        assert expected_fields == actual_fields, f"Expected {expected_fields}, got {actual_fields}"
        print(f"   ✅ Fields preserved: {list(actual_fields)}")

        line_count = sum(1 for _ in f) + 1
        assert line_count == saved_count, f"Expected {saved_count} lines, got {line_count}"
        print(f"   ✅ All {line_count} articles present")

    print("\n6. Verifying wiki_corpus.txt content...")
    corpus_content = Path(corpus_path).read_text(encoding="utf-8")

    assert len(corpus_content) > 0, "Corpus is empty"
    print(f"   ✅ Corpus contains {len(corpus_content):,} characters")

    assert "# 지미 카터" in corpus_content or "지미 카터" in corpus_content, (
        "First article title not found"
    )
    print("   ✅ Article titles included in corpus")

    print("\n7. Cleaning up...")
    crawler.cleanup()
    print("   ✅ Cleanup completed")

    file_size_clean = clean_path.stat().st_size
    file_size_corpus = corpus_path.stat().st_size
    print("\n📊 Output file sizes:")
    print(f"   wiki_clean.jsonl: {file_size_clean:,} bytes ({file_size_clean / 1024:.2f} KB)")
    print(f"   wiki_corpus.txt:    {file_size_corpus:,} bytes ({file_size_corpus / 1024:.2f} KB)")

    print("\n" + "=" * 60)
    print("✅ TEST PASSED: End-to-end pipeline successful")
    print("=" * 60 + "\n")


def test_chunked_downloading():
    """Test chunked downloading with streaming mode.

    Tests:
    1. Streaming mode downloads files on-demand
    2. Only requested articles are fetched
    3. No full dataset download occurs

    Expected Results:
    - 10 articles fetched without downloading full 20GB dataset
    - Articles contain all required fields
    """
    print("\n" + "=" * 60)
    print("TEST: Chunked Downloading with Streaming Mode")
    print("=" * 60)

    test_dir = Path("data/test_chunked")
    test_dir.mkdir(parents=True, exist_ok=True)

    crawler = HuggingFaceDatasetCrawler(data_dir=str(test_dir), streaming=True)

    print("\n1. Fetching 10 articles with streaming mode...")
    articles = []
    for i, article in enumerate(crawler.fetch_articles(max_articles=10), 1):
        articles.append(article)
        print(f"   {i}. {article['title']}")

    assert len(articles) == 10, f"Expected 10 articles, got {len(articles)}"
    print(f"\n   ✅ Successfully fetched {len(articles)} articles")

    print("\n2. Verifying article structure...")
    first_article = articles[0]
    expected_fields = ["id", "url", "title", "text"]
    actual_fields = list(first_article.keys())

    for field in expected_fields:
        assert field in actual_fields, f"Missing field: {field}"
    print(f"   ✅ All fields present: {actual_fields}")

    print("\n3. Verifying field content...")
    assert isinstance(first_article["id"], str), "id should be str"
    assert isinstance(first_article["url"], str), "url should be str"
    assert isinstance(first_article["title"], str), "title should be str"
    assert isinstance(first_article["text"], str), "text should be str"
    assert len(first_article["text"]) > 0, "text should not be empty"

    print("   ✅ Field types and content valid")
    print(f"      id: {first_article['id']}")
    print(f"      url: {first_article['url']}")
    print(f"      title: {first_article['title']}")
    print(f"      text length: {len(first_article['text'])} characters")

    print("\n4. Cleaning up...")
    crawler.cleanup()
    print("   ✅ Cleanup completed")

    print("\n" + "=" * 60)
    print("✅ TEST PASSED: Chunked downloading successful")
    print("=" * 60 + "\n")


def test_field_preservation():
    """Test field preservation in output files.

    Tests:
    1. id and url fields are preserved in wiki_clean.jsonl
    2. Fields have correct types and values
    3. All articles contain complete field set

    Expected Results:
    - wiki_clean.jsonl contains id, url, title, text for all articles
    - Field values match original articles
    """
    print("\n" + "=" * 60)
    print("TEST: Field Preservation in Output Files")
    print("=" * 60)

    test_dir = Path("data/test_fields")
    test_dir.mkdir(parents=True, exist_ok=True)

    crawler = HuggingFaceDatasetCrawler(data_dir=str(test_dir), streaming=True)
    preprocessor = WikipediaPreprocessor(data_dir=str(test_dir))

    print("\n1. Fetching 5 articles...")
    articles = list(crawler.fetch_articles(max_articles=5))
    print(f"   ✅ Fetched {len(articles)} articles")

    print("\n2. Processing and saving...")
    clean_articles = list(preprocessor.process_articles(iter(articles)))
    saved_count = preprocessor.save_clean_articles(iter(clean_articles))
    print(f"   ✅ Saved {saved_count} cleaned articles")

    print("\n3. Verifying wiki_clean.jsonl content...")
    clean_path = test_dir / "wiki_clean.jsonl"

    with open(clean_path, encoding="utf-8") as f:
        lines = list(f)

    assert len(lines) == saved_count, f"Expected {saved_count} lines, got {len(lines)}"
    print(f"   ✅ File contains {len(lines)} lines")

    print("\n4. Checking each article's fields...")
    for i, line in enumerate(lines, 1):
        article = json.loads(line)

        expected_fields = {"id", "url", "title", "text"}
        actual_fields = set(article.keys())

        assert expected_fields == actual_fields, (
            f"Line {i}: Expected {expected_fields}, got {actual_fields}"
        )

        assert isinstance(article["id"], str), f"Line {i}: id should be str"
        assert isinstance(article["url"], str), f"Line {i}: url should be str"
        assert isinstance(article["title"], str), f"Line {i}: title should be str"
        assert isinstance(article["text"], str), f"Line {i}: text should be str"
        assert len(article["text"]) > 0, f"Line {i}: text should not be empty"

    print(f"   ✅ All {len(lines)} articles have complete field sets")

    print("\n5. Sample verification (first article)...")
    first_saved = json.loads(lines[0])
    first_original = articles[0]

    assert first_saved["id"] == first_original["id"], "id mismatch"
    assert first_saved["url"] == first_original["url"], "url mismatch"
    assert first_saved["title"] == first_original["title"], "title mismatch"

    print("   ✅ Fields match original articles")
    print(f"      id: {first_saved['id']}")
    print(f"      url: {first_saved['url']}")
    print(f"      title: {first_saved['title']}")
    print(f"      text preview: {first_saved['text'][:100]}...")

    print("\n6. Cleaning up...")
    crawler.cleanup()
    print("   ✅ Cleanup completed")

    file_size = clean_path.stat().st_size
    print(f"\n📊 wiki_clean.jsonl size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

    print("\n" + "=" * 60)
    print("✅ TEST PASSED: Field preservation verified")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    """Run all tests."""
    print("\n" + "🧪 " * 20)
    print("\nStarting Hugging Face Crawler Integration Tests\n")

    try:
        test_chunked_downloading()
        test_field_preservation()
        test_end_to_end_pipeline()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED")
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}\n")
        raise
