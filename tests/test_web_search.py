"""Unit tests for DuckDuckGo + httpx/bs4 web search — no real network calls."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx
from services.scraper.web_search import WebSearchResult, _extract_text, search_web


# --- _extract_text tests ---


def test_extract_text_strips_scripts_and_nav():
    html = """
    <html><body>
    <script>var x = 1;</script>
    <nav>Menu stuff</nav>
    <div><p>Rosa canina is a wild rose.</p></div>
    <footer>Copyright 2024</footer>
    </body></html>
    """
    text = _extract_text(html)
    assert "Rosa canina" in text
    assert "var x" not in text
    assert "Menu stuff" not in text
    assert "Copyright" not in text


def test_extract_text_collapses_blank_lines():
    html = "<html><body><p>Line 1</p><p></p><p>Line 2</p></body></html>"
    text = _extract_text(html)
    assert text == "Line 1\nLine 2"


def test_extract_text_empty_html():
    assert _extract_text("") == ""


# --- search_web tests ---


DDG_RESULTS = [
    {"title": "Rosa canina - Wikipedia", "href": "https://en.wikipedia.org/wiki/Rosa_canina", "body": "..."},
    {"title": "Dog Rose | PFAF", "href": "https://pfaf.org/rosa_canina", "body": "..."},
]

FAKE_HTML_1 = "<html><body><p>Rosa canina, commonly known as dog rose, is a species of climbing wild rose.</p></body></html>"
FAKE_HTML_2 = "<html><body><p>Rosa canina is edible and has medicinal properties. Known uses include herbal teas.</p></body></html>"


@respx.mock
@pytest.mark.asyncio
async def test_search_web_returns_results():
    """DuckDuckGo search (mocked) + page fetches (mocked) return results."""
    respx.get("https://en.wikipedia.org/wiki/Rosa_canina").mock(
        return_value=httpx.Response(200, text=FAKE_HTML_1)
    )
    respx.get("https://pfaf.org/rosa_canina").mock(
        return_value=httpx.Response(200, text=FAKE_HTML_2)
    )

    with patch("services.scraper.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.text.return_value = DDG_RESULTS
        results = await search_web("Rosa canina")

    assert len(results) == 2
    assert isinstance(results[0], WebSearchResult)
    assert "dog rose" in results[0].content
    assert results[0].url == "https://en.wikipedia.org/wiki/Rosa_canina"
    assert results[1].url == "https://pfaf.org/rosa_canina"


@respx.mock
@pytest.mark.asyncio
async def test_search_web_skips_failed_fetches():
    """Pages that return errors are skipped."""
    respx.get("https://en.wikipedia.org/wiki/Rosa_canina").mock(
        return_value=httpx.Response(200, text=FAKE_HTML_1)
    )
    respx.get("https://pfaf.org/rosa_canina").mock(
        return_value=httpx.Response(500)
    )

    with patch("services.scraper.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.text.return_value = DDG_RESULTS
        results = await search_web("Rosa canina")

    assert len(results) == 1
    assert results[0].url == "https://en.wikipedia.org/wiki/Rosa_canina"


@respx.mock
@pytest.mark.asyncio
async def test_search_web_skips_short_content():
    """Pages with less than 50 chars of text are skipped."""
    respx.get("https://en.wikipedia.org/wiki/Rosa_canina").mock(
        return_value=httpx.Response(200, text="<html><body><p>Short</p></body></html>")
    )

    with patch("services.scraper.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.text.return_value = [DDG_RESULTS[0]]
        results = await search_web("Rosa canina")

    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_web_disabled():
    """Returns empty list when web search is disabled."""
    with patch("services.scraper.web_search.settings") as mock_settings:
        mock_settings.web_search_enabled = False
        results = await search_web("Rosa canina")

    assert results == []


@pytest.mark.asyncio
async def test_search_web_ddg_exception():
    """Returns empty list when DuckDuckGo raises an exception."""
    with patch("services.scraper.web_search.DDGS") as MockDDGS:
        MockDDGS.return_value.text.side_effect = Exception("rate limited")
        results = await search_web("Rosa canina")

    assert results == []
