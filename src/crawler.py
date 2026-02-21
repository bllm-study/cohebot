import os
import bz2
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin

import requests
from tqdm import tqdm


class WikipediaDumpCrawler:

    DUMP_BASE_URL = "https://dumps.wikimedia.org/kowiki/latest/"
    DUMP_FILENAME = "kowiki-latest-pages-articles.xml.bz2"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dump_path = self.data_dir / self.DUMP_FILENAME

    def download_dump(self, force: bool = False) -> Path:
        if self.dump_path.exists() and not force:
            print(f"덤프 파일이 이미 존재합니다: {self.dump_path}")
            return self.dump_path

        url = urljoin(self.DUMP_BASE_URL, self.DUMP_FILENAME)
        print(f"다운로드 시작: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(self.dump_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc="다운로드") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"다운로드 완료: {self.dump_path}")
        return self.dump_path

    def parse_dump(self, max_articles: int | None = None) -> Iterator[dict]:
        if not self.dump_path.exists():
            raise FileNotFoundError(f"덤프 파일이 없습니다: {self.dump_path}")

        print(f"덤프 파싱 시작: {self.dump_path}")

        count = 0
        title = None
        text = None
        in_page = False

        with bz2.open(self.dump_path, "rt", encoding="utf-8") as f:
            for event, elem in ET.iterparse(f, events=("start", "end")):
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

                if event == "start" and tag == "page":
                    in_page = True
                    title = None
                    text = None

                elif event == "end":
                    if tag == "title" and in_page:
                        title = elem.text or ""

                    elif tag == "text" and in_page:
                        text = elem.text or ""

                    elif tag == "page" and in_page:
                        in_page = False

                        if title and text:
                            text_lower = text.strip().lower()
                            if text_lower.startswith("#redirect") or \
                               text_lower.startswith("#넘겨주기"):
                                elem.clear()
                                continue

                            if ":" in title:
                                prefix = title.split(":")[0]
                                if prefix in [
                                    "위키백과", "Wikipedia", "파일", "File",
                                    "틀", "Template", "분류", "Category",
                                    "포털", "Portal", "모듈", "Module",
                                    "사용자", "User", "토론", "Talk",
                                    "미디어위키", "MediaWiki", "도움말", "Help"
                                ]:
                                    elem.clear()
                                    continue

                            yield {"title": title, "text": text}
                            count += 1

                            if count % 1000 == 0:
                                print(f"  {count}개 문서 파싱됨...")

                            if max_articles and count >= max_articles:
                                print(f"파싱 완료: {count}개 문서")
                                return

                        elem.clear()

        print(f"파싱 완료: {count}개 문서")


class WikipediaAPICrawler:

    API_URL = "https://ko.wikipedia.org/w/api.php"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "GPT2-Study-Bot/1.0 (Educational Purpose)"
        })

    def get_random_articles(self, count: int = 100) -> Iterator[dict]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": min(count, 50),
            "prop": "extracts",
            "explaintext": True,
            "exlimit": "max",
        }

        fetched = 0
        while fetched < count:
            response = self.session.get(self.API_URL, params=params)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if "extract" in page:
                    yield {
                        "title": page.get("title", ""),
                        "text": page.get("extract", "")
                    }
                    fetched += 1
                    if fetched >= count:
                        break

    def get_articles_by_category(
        self,
        category: str,
        max_articles: int = 1000
    ) -> Iterator[dict]:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"분류:{category}",
            "cmtype": "page",
            "cmlimit": "max",
        }

        titles = []
        while len(titles) < max_articles:
            response = self.session.get(self.API_URL, params=params)
            data = response.json()

            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                titles.append(member["title"])
                if len(titles) >= max_articles:
                    break

            if "continue" in data:
                params["cmcontinue"] = data["continue"]["cmcontinue"]
            else:
                break

        for i in tqdm(range(0, len(titles), 50), desc=f"'{category}' 크롤링"):
            batch = titles[i:i + 50]
            params = {
                "action": "query",
                "format": "json",
                "titles": "|".join(batch),
                "prop": "extracts",
                "explaintext": True,
                "exlimit": "max",
            }

            response = self.session.get(self.API_URL, params=params)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if "extract" in page:
                    yield {
                        "title": page.get("title", ""),
                        "text": page.get("extract", "")
                    }

    def search_articles(
        self,
        query: str,
        max_articles: int = 100
    ) -> Iterator[dict]:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": min(max_articles, 50),
            "srprop": "snippet",
        }

        response = self.session.get(self.API_URL, params=params)
        data = response.json()

        titles = [r["title"] for r in data.get("query", {}).get("search", [])]

        for title in tqdm(titles, desc=f"'{query}' 검색 결과"):
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            }

            response = self.session.get(self.API_URL, params=params)
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if "extract" in page:
                    yield {
                        "title": page.get("title", ""),
                        "text": page.get("extract", "")
                    }


def download_korean_wikipedia(
    data_dir: str = "data",
    max_articles: int | None = None
) -> list[dict]:
    crawler = WikipediaDumpCrawler(data_dir)

    crawler.download_dump()

    articles = list(tqdm(
        crawler.parse_dump(max_articles),
        desc="문서 로딩",
        total=max_articles
    ))

    return articles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="한국어 위키피디아 크롤러")
    parser.add_argument(
        "--method",
        choices=["dump", "api"],
        default="dump",
        help="크롤링 방법 (dump: 전체 덤프, api: API 사용)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="최대 문서 수"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="데이터 저장 디렉토리"
    )
    args = parser.parse_args()

    if args.method == "dump":
        articles = download_korean_wikipedia(
            data_dir=args.data_dir,
            max_articles=args.max_articles
        )
    else:
        crawler = WikipediaAPICrawler(args.data_dir)
        articles = list(crawler.get_random_articles(args.max_articles or 100))

    print(f"\n총 {len(articles)}개 문서 수집")
    if articles:
        print(f"\n예시 문서: {articles[0]['title']}")
        print(articles[0]["text"][:500] + "...")
