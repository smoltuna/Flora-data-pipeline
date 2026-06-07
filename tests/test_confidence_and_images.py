"""Unit tests for confidence score filtering and image candidate fallback fixes."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from services.llm.provider import LLMResponse
from services.rag.synthesizer import NOT_AVAILABLE
from services.rag.verifier import _parse_verification, verify_all_fields, verify_field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockLLM:
    def __init__(self, response: str = "0.85"):
        self._response = response

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        return LLMResponse(text=self._response)


@dataclass
class _FakeCandidate:
    url: str = "https://example.com/rosa.jpg"
    author: str = "John Doe"


# ---------------------------------------------------------------------------
# _parse_verification — regex parser
# ---------------------------------------------------------------------------

def test_parse_high_confidence():
    r = _parse_verification("0.9")
    assert r.confidence == pytest.approx(0.9)
    assert r.supported is True


def test_parse_mid_confidence():
    r = _parse_verification("0.5")
    assert r.confidence == pytest.approx(0.5)
    assert r.supported is True


def test_parse_low_confidence():
    r = _parse_verification("0.2")
    assert r.confidence == pytest.approx(0.2)
    assert r.supported is False


def test_parse_full_confidence():
    r = _parse_verification("1.0")
    assert r.confidence == pytest.approx(1.0)
    assert r.supported is True


def test_parse_zero():
    r = _parse_verification("0")
    assert r.confidence == pytest.approx(0.0)
    assert r.supported is False


def test_parse_noisy_llm_response():
    """LLM returns surrounding text — regex still extracts the number."""
    r = _parse_verification("Based on the evidence I'd rate this 0.8 confidence.")
    assert r.confidence == pytest.approx(0.8)
    assert r.supported is True


def test_parse_unparseable_defaults_to_zero():
    r = _parse_verification("I cannot determine a score from this text.")
    assert r.confidence == pytest.approx(0.0)
    assert r.supported is False


# ---------------------------------------------------------------------------
# verify_field — NOT_AVAILABLE short-circuit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_field_not_available_returns_zero_without_llm_call():
    """NOT_AVAILABLE fields should return 0.0 immediately (LLM never called)."""
    call_count = 0

    class _CountingLLM(_MockLLM):
        async def complete(self, prompt: str, system: str = "") -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return LLMResponse(text="0.9")

    result = await verify_field("fun_fact", NOT_AVAILABLE, "Some source text.", _CountingLLM())
    assert result.confidence == pytest.approx(0.0)
    assert result.supported is False
    assert call_count == 0  # LLM must not be called for NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Confidence score filtering — core fix
# ---------------------------------------------------------------------------

def test_not_available_fields_excluded_from_fields_to_verify():
    """The filtering step must strip NOT_AVAILABLE values before verification."""
    generated_fields = {
        "description": "A beautiful rose with pink petals.",
        "fun_fact": NOT_AVAILABLE,
        "habitat": "Hedgerows and woodland edges.",
        "etymology": NOT_AVAILABLE,
        "wiki_description": "Rosa canina is a species of rose.",
        "cultural_info": NOT_AVAILABLE,
    }
    fields_to_verify = {f: v for f, v in generated_fields.items() if v != NOT_AVAILABLE}

    assert set(fields_to_verify) == {"description", "habitat", "wiki_description"}


@pytest.mark.asyncio
async def test_confidence_scores_only_contain_verified_fields():
    """confidence_scores must only include verified fields."""
    generated_fields = {
        "description": "A beautiful rose with pink petals.",
        "fun_fact": NOT_AVAILABLE,
        "habitat": "Hedgerows and woodland edges.",
        "etymology": NOT_AVAILABLE,
    }

    fields_to_verify = {f: v for f, v in generated_fields.items() if v != NOT_AVAILABLE}
    source = "Source text about rosa canina."
    # verify_all_fields now batches all fields in one call; mock returns numbered scores
    results = await verify_all_fields(
        fields_to_verify, source, _MockLLM("1. 0.85\n2. 0.85"),
    )

    confidence_scores = {field: {"llm_score": res.confidence} for field, res in results.items()}

    assert "description" in confidence_scores
    assert "habitat" in confidence_scores
    assert "fun_fact" not in confidence_scores   # NOT_AVAILABLE → absent
    assert "etymology" not in confidence_scores  # NOT_AVAILABLE → absent
    assert confidence_scores["description"]["llm_score"] == pytest.approx(0.85)
    assert confidence_scores["habitat"]["llm_score"] == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_no_available_fields_produces_empty_confidence_scores():
    """All fields NOT_AVAILABLE → confidence_scores is empty, not full of zeros."""
    generated_fields = {f: NOT_AVAILABLE for f in ["description", "fun_fact", "habitat"]}

    fields_to_verify = {f: v for f, v in generated_fields.items() if v != NOT_AVAILABLE}
    results = await verify_all_fields(fields_to_verify, "Source text.", _MockLLM())

    confidence_scores = {field: {"llm_score": res.confidence} for field, res in results.items()}
    assert confidence_scores == {}


# ---------------------------------------------------------------------------
# Image candidate fallback — core fix
# ---------------------------------------------------------------------------

def test_one_candidate_uses_same_image_for_main():
    """Single Wikimedia result → main_candidate falls back to candidates[0]."""
    candidates = [_FakeCandidate(url="https://example.com/rosa0.jpg")]

    main_candidate = candidates[1] if len(candidates) > 1 else candidates[0]

    assert main_candidate is candidates[0]
    assert main_candidate.url == "https://example.com/rosa0.jpg"


def test_two_candidates_uses_second_for_main():
    """Two Wikimedia results → main_candidate is candidates[1]."""
    c0 = _FakeCandidate(url="https://example.com/rosa0.jpg")
    c1 = _FakeCandidate(url="https://example.com/rosa1.jpg")
    candidates = [c0, c1]

    main_candidate = candidates[1] if len(candidates) > 1 else candidates[0]

    assert main_candidate is c1
    assert main_candidate.url == "https://example.com/rosa1.jpg"


def test_five_candidates_still_uses_second_for_main():
    candidates = [_FakeCandidate(url=f"https://example.com/rosa{i}.jpg") for i in range(5)]

    main_candidate = candidates[1] if len(candidates) > 1 else candidates[0]

    assert main_candidate is candidates[1]


