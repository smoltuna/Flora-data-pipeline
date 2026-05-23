"""Unit tests for RecursiveTextSplitter."""
import pytest

from services.rag.chunker import RecursiveTextSplitter


def _word_count(text: str) -> int:
    return len(text.split())


def test_split_short_text_single_chunk() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100)
    short = "This is a short text with very few words."
    result = splitter.split(short)
    assert len(result) == 1
    assert result[0] == short


def test_split_long_text_paragraph_boundary() -> None:
    # Build text with clear paragraph breaks
    para = "Word " * 80  # ~80 words per paragraph
    text = (para.strip() + "\n\n") * 5  # 5 paragraphs, ~400 words each
    splitter = RecursiveTextSplitter(chunk_size=200, chunk_overlap=20)
    result = splitter.split(text)
    # Should split — total words ~400 which exceeds chunk_size=200
    assert len(result) > 1
    # The first separator tried is "\n\n", so splits should happen at paragraph boundaries
    for chunk in result:
        assert "\n\n" not in chunk.strip() or _word_count(chunk) <= 200 * 1.1


def test_split_respects_chunk_size() -> None:
    # 1000-word text, chunk_size=100
    words = ["word"] * 1000
    text = " ".join(words)
    splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=10)
    result = splitter.split(text)
    assert len(result) > 1
    for chunk in result:
        # Allow 10% overflow (word boundary rounding)
        assert _word_count(chunk) <= 100 * 1.1, f"Chunk too large: {_word_count(chunk)} words"


def test_overlap() -> None:
    # Build text that forces multiple chunks and check overlap
    # 600 words at chunk_size=200, overlap=50 — we expect ~3-4 chunks
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words)
    splitter = RecursiveTextSplitter(chunk_size=200, chunk_overlap=50)
    result = splitter.split(text)
    assert len(result) >= 2

    # Check that consecutive chunks share some words (overlap)
    for i in range(len(result) - 1):
        words_a = set(result[i].split())
        words_b = set(result[i + 1].split())
        shared = words_a & words_b
        assert len(shared) > 0, f"No overlap between chunk {i} and chunk {i+1}"


def test_empty_text_returns_empty() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100)
    assert splitter.split("") == []
    assert splitter.split("   ") == []


def test_single_word_text() -> None:
    splitter = RecursiveTextSplitter(chunk_size=100)
    result = splitter.split("hello")
    assert result == ["hello"]
