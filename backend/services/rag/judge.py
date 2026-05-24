"""LLM-as-Judge — scores generated text fields beyond Self-RAG's source-support check.

Self-RAG asks "is this claim grounded?". The judge asks "is this output good for a
user?" — covering factual accuracy, completeness, coherence, source fidelity, and
engagement. One LLM call per populated field; scores are stored alongside Self-RAG
confidence in flowers.confidence_scores.
"""
from __future__ import annotations

import re

import structlog
from config import settings

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

log = structlog.get_logger()

NOT_AVAILABLE = "Information not available."

EVAL_CRITERIA: dict[str, str] = {
    "factual_accuracy": "Are the stated facts correct and verifiable from the sources?",
    "completeness":     "Does the text cover the most important aspects of this topic?",
    "coherence":        "Is the text well-written, clear, and natural-sounding?",
    "source_fidelity":  "Does the text faithfully represent the sources without hallucination?",
    "engagement":       "Is the text interesting and suitable for a mobile app audience?",
}


def _format_sources(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
    """Concatenate chunk texts with source labels, capped at max_chars."""
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        text = f"[{chunk.source}] {chunk.chunk_text}"
        if total + len(text) > max_chars:
            text = text[: max(0, max_chars - total)]
            if text:
                parts.append(text)
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts) if parts else "(no sources)"


def _parse_scores(response: str) -> dict[str, float]:
    """Parse '<criterion>: <0.0-1.0>' lines from the LLM response.

    Tolerant of extra commentary — only criterion names from EVAL_CRITERIA are picked
    up, and any value is clamped to [0.0, 1.0]. Missing criteria default to 0.0.
    """
    scores: dict[str, float] = {}
    for criterion in EVAL_CRITERIA:
        pattern = rf"{re.escape(criterion)}\s*[:=]\s*(0?\.\d+|1\.0|[01])"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                scores[criterion] = max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                scores[criterion] = 0.0
        else:
            scores[criterion] = 0.0
    return scores


async def judge_field(
    field_name: str,
    field_value: str,
    latin_name: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
) -> dict[str, float]:
    """Score a single field on EVAL_CRITERIA. Returns {criterion: score}."""
    sources_block = _format_sources(chunks)
    criteria_block = "\n".join(
        f"- {name}: {desc}" for name, desc in EVAL_CRITERIA.items()
    )
    criteria_list = ", ".join(EVAL_CRITERIA.keys())

    response = await llm.complete(
        prompt=(
            f'You are evaluating the "{field_name}" field of an enriched record for '
            f"the plant {latin_name}.\n\n"
            f"GENERATED TEXT:\n{field_value}\n\n"
            f"SOURCE MATERIAL:\n{sources_block}\n\n"
            f"Score the GENERATED TEXT on these criteria (0.0 to 1.0):\n"
            f"{criteria_block}\n\n"
            f"Reply with one line per criterion in the format:\n"
            f"<criterion>: <score>\n"
            f"Use exactly these criterion names: {criteria_list}.\n"
            f"Return ONLY the scores, no explanation."
        ),
        system="You are a strict, concise evaluator. Output only criterion scores.",
    )
    return _parse_scores(response.text)


async def judge_flower(
    latin_name: str,
    field_values: dict[str, str],
    field_chunks: dict[str, list[RetrievedChunk]],
    llm: LLMProvider,
) -> dict[str, dict[str, float]]:
    """Score each populated text field on EVAL_CRITERIA.

    Fields whose value is empty or NOT_AVAILABLE are skipped (no judge call).
    Returns {field_name: {criterion: score}}.

    Logs a warning when factual_accuracy < quality_gate_threshold — informational
    only, the field is not modified.
    """
    results: dict[str, dict[str, float]] = {}
    threshold = settings.quality_gate_threshold

    for field_name, value in field_values.items():
        if not value or value == NOT_AVAILABLE:
            continue
        chunks = field_chunks.get(field_name, [])
        try:
            scores = await judge_field(field_name, value, latin_name, chunks, llm)
        except Exception as exc:
            log.warning(
                "judge.field_failed",
                field=field_name,
                latin_name=latin_name,
                error=str(exc),
            )
            continue

        results[field_name] = scores
        if scores.get("factual_accuracy", 0.0) < threshold:
            log.warning(
                "judge.quality_gate_failed",
                field=field_name,
                latin_name=latin_name,
                factual_accuracy=scores.get("factual_accuracy", 0.0),
                threshold=threshold,
            )
    return results
