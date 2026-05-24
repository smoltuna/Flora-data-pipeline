"""Unit tests for LLM-as-Judge evaluation."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.llm.provider import LLMResponse
from services.rag.judge import EVAL_CRITERIA, NOT_AVAILABLE, judge_flower
from services.rag.retriever import RetrievedChunk


class _MockLLM:
    """Returns a canned response for every call."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        self.calls.append((prompt, system))
        return LLMResponse(text=self.response_text, tokens_used=0)


def _chunk(text: str = "petal data", source: str = "wikipedia") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1, chunk_text=text, source=source, rrf_score=1.0, embedding=[]
    )


def _full_score_block(value: float = 0.9) -> str:
    return "\n".join(f"{name}: {value}" for name in EVAL_CRITERIA)


@pytest.mark.asyncio
async def test_judge_return_schema():
    """judge_flower returns dict[str, dict[str, float]] with all 5 criteria per field."""
    llm = _MockLLM(_full_score_block(0.85))
    field_values = {"description": "A spring perennial.", "fun_fact": "Rare blue variety."}
    field_chunks = {
        "description": [_chunk()],
        "fun_fact": [_chunk("rare variety found in Japan")],
    }

    result = await judge_flower("Iris germanica", field_values, field_chunks, llm)

    assert set(result.keys()) == {"description", "fun_fact"}
    for field_scores in result.values():
        assert set(field_scores.keys()) == set(EVAL_CRITERIA.keys())
        for score in field_scores.values():
            assert isinstance(score, float)


@pytest.mark.asyncio
async def test_judge_scores_in_range():
    """All returned scores are clamped to [0.0, 1.0] regardless of LLM output."""
    # LLM returns out-of-range values; parser should clamp them
    bad_response = "\n".join([
        "factual_accuracy: 0.7",
        "completeness: 0.5",
        "coherence: 1.0",
        "source_fidelity: 0.8",
        "engagement: 0.6",
    ])
    llm = _MockLLM(bad_response)
    result = await judge_flower(
        "Iris germanica",
        {"description": "A flower."},
        {"description": [_chunk()]},
        llm,
    )

    for scores in result.values():
        for score in scores.values():
            assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_judge_skips_unavailable_fields():
    """Fields with empty value or NOT_AVAILABLE are not judged (no LLM call)."""
    llm = _MockLLM(_full_score_block(0.9))
    field_values = {
        "description": "A spring perennial.",
        "etymology": NOT_AVAILABLE,
        "fun_fact": "",
    }
    field_chunks = {"description": [_chunk()]}

    result = await judge_flower("Iris germanica", field_values, field_chunks, llm)

    assert "description" in result
    assert "etymology" not in result
    assert "fun_fact" not in result
    assert len(llm.calls) == 1  # only description triggered a call
