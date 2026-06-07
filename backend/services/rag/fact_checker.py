"""Post-synthesis web fact-check for COMPLEX fields.

CRAG (services/rag/grader.py) only fires before synthesis and only when retrieval
is graded insufficient. It can't catch the case where the synthesizer hallucinates
over a correct chunk — see Tulipa gesneriana etymology, where Wikipedia clearly
says "Conrad Gesner" but llama3.2:3b wrote "Joannes Andreae Gesner (1566-1619)".

This module fills that gap. For each COMPLEX field (etymology, cultural_info,
fun_fact), it runs a targeted DuckDuckGo search, asks a small LLM whether the
synthesized claim agrees with the top snippet, and regenerates the field with
the snippet pinned in context if they disagree. One regen per field, max.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from services.llm.provider import LLMProvider
from services.rag.synthesizer import NOT_AVAILABLE

log = structlog.get_logger()

# COMPLEX fields most likely to suffer hallucination; matches FIELD_CONFIG.
CHECKABLE_FIELDS = ("etymology", "cultural_info", "fun_fact")

# Targeted-query hints per field. Generic enough to surface authoritative pages.
_QUERY_HINT = {
    "etymology": "{name} etymology meaning name origin",
    "cultural_info": "{name} history culture symbolism",
    "fun_fact": "{name} botany interesting facts",
}

# How much snippet to feed the comparison LLM. ~1500 chars ≈ ~400 tokens.
_SNIPPET_CHARS = 1500


@dataclass
class FactCheckResult:
    field: str
    agreed: bool | None  # None = inconclusive (no snippet, LLM error)
    original: str
    corrected: str | None  # None if no regen happened
    snippet_url: str | None


async def fact_check_fields(
    latin_name: str,
    common_name: str | None,
    synthesized: dict[str, str],
    llm: LLMProvider,
) -> dict[str, FactCheckResult]:
    """Run a web fact-check for each COMPLEX field; regenerate disagreeing ones.

    Returns {field: FactCheckResult}. Caller is responsible for writing the
    corrected values back to the Flower row.
    """
    from config import settings

    from services.scraper.web_search import search_web

    if not settings.fact_check_enabled:
        return {}

    results: dict[str, FactCheckResult] = {}
    display_name = common_name or latin_name

    for field_name in CHECKABLE_FIELDS:
        original = synthesized.get(field_name)
        if not original or original == NOT_AVAILABLE:
            continue

        hint = _QUERY_HINT[field_name].format(name=latin_name)
        try:
            hits = await search_web(latin_name, common_name, max_results=2, query=hint)
        except Exception as e:
            log.warning("fact_check.search_failed", field=field_name, error=str(e))
            continue

        if not hits:
            log.debug("fact_check.no_results", field=field_name)
            continue

        snippet = hits[0].content[:_SNIPPET_CHARS]
        snippet_url = hits[0].url

        agreed = await _compare_claim_to_snippet(
            field_name, original, snippet, latin_name, llm,
        )

        if agreed is False:
            corrected = await _regenerate_with_snippet(
                field_name, latin_name, display_name, snippet, llm,
            )
            if corrected and corrected != NOT_AVAILABLE:
                log.info(
                    "fact_check.regenerated",
                    field=field_name, latin_name=latin_name,
                    snippet_url=snippet_url,
                )
                results[field_name] = FactCheckResult(
                    field=field_name, agreed=False,
                    original=original, corrected=corrected, snippet_url=snippet_url,
                )
                continue

        results[field_name] = FactCheckResult(
            field=field_name, agreed=agreed,
            original=original, corrected=None, snippet_url=snippet_url,
        )

    return results


async def _compare_claim_to_snippet(
    field_name: str,
    claim: str,
    snippet: str,
    latin_name: str,
    llm: LLMProvider,
) -> bool | None:
    """Ask the LLM whether the claim contradicts the snippet. Returns True/False/None."""
    try:
        resp = await llm.complete(
            prompt=(
                f"For the plant {latin_name}, is the CLAIM consistent with the SOURCE?\n\n"
                f"CLAIM ({field_name}): {claim}\n\n"
                f"SOURCE:\n{snippet}\n\n"
                f"Reply with exactly one word:\n"
                f"- AGREE if the source supports the claim\n"
                f"- DISAGREE if the source contradicts the claim "
                f"(wrong name, wrong date, wrong place, wrong meaning)\n"
                f"- UNCLEAR if the source does not address the claim\n"
            ),
            system="You are a strict fact-checker. Reply with exactly one word.",
        )
    except Exception as e:
        log.warning("fact_check.compare_failed", field=field_name, error=str(e))
        return None

    text = resp.text.strip().upper()
    # Take the first verdict word — models sometimes add commentary
    match = re.search(r"\b(AGREE|DISAGREE|UNCLEAR)\b", text)
    if not match:
        return None
    verdict = match.group(1)
    if verdict == "DISAGREE":
        return False
    if verdict == "AGREE":
        return True
    return None


async def _regenerate_with_snippet(
    field_name: str,
    latin_name: str,
    display_name: str,
    snippet: str,
    llm: LLMProvider,
) -> str | None:
    """Re-synthesize one field with the authoritative snippet pinned as the only source."""
    field_prompts = {
        "etymology": (
            f"Write a complete one-sentence etymology for {display_name} "
            f"({latin_name}) — state the origin and meaning of the Latin name."
        ),
        "cultural_info": (
            f"Write 1-2 complete sentences on the cultural or historical "
            f"significance of {display_name} ({latin_name})."
        ),
        "fun_fact": (
            f"Write one complete surprising fact about {display_name} "
            f"({latin_name})."
        ),
    }
    instruction = field_prompts.get(field_name)
    if not instruction:
        return None

    try:
        resp = await llm.complete(
            prompt=(
                f"{instruction}\n\n"
                f"SOURCE:\n{snippet}\n\n"
                f"Rules:\n"
                f"- Ground every claim in the source. Do not invent facts.\n"
                f"- Write a complete sentence with subject and verb — never a "
                f"single word or fragment.\n"
                f'- If the source does not support a confident answer, reply exactly: '
                f'"{NOT_AVAILABLE}"\n'
                f"- Reply with only the field value, no preamble."
            ),
            system="You are a precise botanical writer. Output only the field value.",
        )
    except Exception as e:
        log.warning("fact_check.regen_failed", field=field_name, error=str(e))
        return None

    text = resp.text.strip().strip("\"'")
    if not text:
        return None
    # Reject suspiciously short regens — a single word like "Melanium" is
    # almost certainly the model returning a fragment instead of a sentence.
    if len(text.split()) < 4:
        log.info("fact_check.regen_too_short", field=field_name, preview=text[:80])
        return None
    return text
