"""DuckDuckGo web search + httpx/BeautifulSoup scraper — 5th unstructured source."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from config import settings
from ddgs import DDGS


@dataclass
class WebSearchResult:
    url: str
    title: str
    content: str
    score: float = 0.0


def _extract_text(html: str) -> str:
    """Extract visible text from HTML, stripping nav/script/style noise."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


async def search_web(
    latin_name: str,
    common_name: str | None = None,
    max_results: int = 5,
    query: str | None = None,
) -> list[WebSearchResult]:
    """Search DuckDuckGo for the flower, then scrape each result page.

    Returns up to max_results results with extracted clean text.
    Returns [] if web search is disabled.
    Pass `query` to override the default "{latin_name} flower" search term.
    """
    if not settings.web_search_enabled:
        return []

    if not query:
        query = f"{latin_name} flower"
        if common_name:
            query = f"{latin_name} {common_name} flower"

    # DuckDuckGo search is synchronous — run in thread to avoid blocking
    loop = asyncio.get_running_loop()
    try:
        hits = await loop.run_in_executor(
            None, lambda: DDGS().text(query, max_results=max_results)
        )
    except Exception:
        return []

    if not hits:
        return []

    # Fetch all result pages concurrently
    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FloraBot/1.0)"},
    ) as client:
        tasks = [client.get(hit["href"]) for hit in hits]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[WebSearchResult] = []
    for hit, resp in zip(hits, responses):
        if isinstance(resp, BaseException) or resp.status_code != 200:
            continue
        text = _extract_text(resp.text)
        if len(text) < 50:
            continue
        # Truncate very long pages to keep chunk sizes reasonable
        if len(text) > 8000:
            text = text[:8000]
        results.append(
            WebSearchResult(
                url=hit["href"],
                title=hit.get("title", ""),
                content=text,
                score=0.0,
            )
        )

    return results
