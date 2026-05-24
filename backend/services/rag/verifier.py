"""Self-RAG verification — checks that generated field values are grounded in sources.

Asks the LLM to cite the supporting passage for each generated field.
Returns a confidence score (0–1) stored as confidence_scores JSONB in the flowers table.
Source confidence weighting adjusts scores based on how reliable the supporting sources are.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

# Reliability weights per source (higher = more trustworthy)
SOURCE_CONFIDENCE: dict[str, float] = {
    "pfaf": 0.95,
    "wikidata": 0.90,
    "gbif": 0.90,
    "wikipedia": 0.80,
    "web_ddg": 0.65,
    "web_ddg_correction": 0.65,
}


def _source_weight(chunks: list[RetrievedChunk]) -> float:
    """Return the max source confidence across chunks (best source sets the weight)."""
    if not chunks:
        return 1.0
    weights: list[float] = []
    for c in chunks:
        for src_key, w in SOURCE_CONFIDENCE.items():
            if c.source == src_key or c.source.startswith(src_key + "_"):
                weights.append(w)
                break
        else:
            weights.append(0.70)  # unknown source gets moderate weight
    return max(weights)


class VerificationResult(BaseModel):
    supported: bool = False
    quote: str = "none"
    confidence: float = 0.0


async def verify_field(
    field_name: str,
    field_value: str,
    source_text: str,
    llm: LLMProvider,
) -> VerificationResult:
    """Verify that a generated field value is supported by the source material."""
    if not field_value or field_value == "Information not available.":
        return VerificationResult(supported=False, quote="none", confidence=0.0)

    response = await llm.complete(
        prompt=f"""Rate how well this claim is supported by the source material.

Claim: {field_name} = "{field_value[:300]}"

Source material:
{source_text[:5000]}

Reply with a single decimal number from 0.0 to 1.0:
- 1.0 = claim is explicitly stated in the source
- 0.5 = claim is partially supported or implied
- 0.0 = claim is not found in the source

Reply with ONLY the number, nothing else.""",
        system="You are a fact-checking assistant. Reply with only a decimal number.",
    )

    return _parse_verification(response.text)


async def verify_all_fields(
    generated_fields: dict[str, str],
    source_text: str,
    llm: LLMProvider,
    field_chunks: dict[str, list[RetrievedChunk]] | None = None,
) -> dict[str, VerificationResult]:
    """Verify all generated text fields against source text in a single LLM call.

    If field_chunks is provided, LLM scores are scaled by the max source confidence
    for each field's contributing chunks (e.g. web-only fields score lower than
    PFAF-backed fields regardless of LLM judgment).
    """
    # Filter out empty / unavailable fields
    to_verify = {
        f: v for f, v in generated_fields.items()
        if v and v != "Information not available."
    }
    if not to_verify:
        return {f: VerificationResult() for f in generated_fields}

    # Build a numbered claims block
    field_list = list(to_verify.keys())
    claims = "\n".join(
        f"{i+1}. {name} = \"{value[:300]}\""
        for i, (name, value) in enumerate(to_verify.items())
    )

    response = await llm.complete(
        prompt=f"""Rate how well each claim below is supported by the source material.

Claims:
{claims}

Source material:
{source_text[:5000]}

For each claim, reply with its number and a score from 0.0 to 1.0:
- 1.0 = explicitly stated in the source
- 0.5 = partially supported or implied
- 0.0 = not found in the source

Reply with ONLY numbered scores, one per line, like:
1. 0.8
2. 0.5""",
        system="You are a fact-checking assistant. Reply with only numbered scores.",
    )

    # Parse numbered scores from response (e.g. "1. 0.8\n2. 0.5")
    response_text = response.text
    score_map: dict[int, float] = {}
    for match in re.finditer(r"(\d+)\D*(0?\.\d+|1\.0|[01])\b", response_text):
        idx = int(match.group(1))
        if 1 <= idx <= len(field_list):
            score = max(0.0, min(1.0, float(match.group(2))))
            score_map[idx] = score

    # Fallback: if no numbered scores found, parse bare floats in order
    if not score_map:
        bare_scores = re.findall(r"(0?\.\d+|1\.0|[01])\b", response_text.strip())
        for i, s in enumerate(bare_scores[:len(field_list)]):
            score_map[i + 1] = max(0.0, min(1.0, float(s)))

    results: dict[str, VerificationResult] = {}
    for i, field_name in enumerate(field_list):
        raw_score = score_map.get(i + 1, 0.0)

        # Apply source confidence weighting if chunk info is available
        if field_chunks and field_name in field_chunks:
            weight = _source_weight(field_chunks[field_name])
            score = raw_score * weight
        else:
            score = raw_score

        results[field_name] = VerificationResult(
            supported=score >= 0.5, confidence=round(score, 4),
        )

    # Add empty results for fields that were skipped
    for f in generated_fields:
        if f not in results:
            results[f] = VerificationResult()

    return results


def _parse_verification(response: str) -> VerificationResult:
    text = response.strip()
    match = re.search(r"\b(0?\.\d+|1\.0|[01])\b", text)
    if match:
        try:
            score = max(0.0, min(1.0, float(match.group())))
            return VerificationResult(supported=score >= 0.5, confidence=score)
        except ValueError:
            pass
    return VerificationResult()
