"""LLM synthesis — constructs prompts from per-field context and generates structured output.

Accepts either raw chunks (SIMPLE fields) or pre-extracted facts (COMPLEX fields)
per field. Output is validated against Pydantic v2 schemas.
"""
from __future__ import annotations

import json
import re

import structlog
from pydantic import BaseModel, Field, ValidationError

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

log = structlog.get_logger()

NOT_AVAILABLE = "Information not available."

# Per-field context budget. ~600 tokens × ~5 active fields ≈ 3000 tokens of source
# material in the prompt — matches the v1 budget. Tokens ≈ words × 1.3 for English,
# so 600 tokens ≈ 460 words. Use 450 to stay under.
_MAX_FIELD_CONTEXT_WORDS = 450

# Source display labels for attribution in prompts
_SOURCE_LABELS = {
    "pfaf": "PFAF (Plants For A Future)",
    "wikipedia": "Wikipedia",
    "wikidata": "Wikidata",
    "gbif": "GBIF",
}


def _truncate_words(text: str, max_words: int) -> tuple[str, bool]:
    """Truncate text to max_words. Returns (text, was_truncated)."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]) + " …", True


class SynthesizedFlower(BaseModel):
    description: str = Field(default=NOT_AVAILABLE)
    fun_fact: str = Field(default=NOT_AVAILABLE)
    wiki_description: str = Field(default=NOT_AVAILABLE)
    habitat: str = Field(default=NOT_AVAILABLE)
    etymology: str = Field(default=NOT_AVAILABLE)
    cultural_info: str = Field(default=NOT_AVAILABLE)
    petal_color_hex: str | None = Field(default=None)
    care_info: dict | list = Field(default_factory=dict)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Group chunks by source for attributed context."""
    by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.source, []).append(chunk.chunk_text)

    sections: list[str] = []
    for source, texts in by_source.items():
        label = _SOURCE_LABELS.get(source, source.upper())
        sections.append(f"[{label}]\n" + "\n---\n".join(texts))
    return "\n\n".join(sections)


def _format_per_field_context(
    field_context: dict[str, list[RetrievedChunk] | str],
) -> str:
    """Build a per-field context block for the synthesis prompt.

    SIMPLE fields → grouped-by-source chunk text.
    COMPLEX fields → pre-extracted fact strings.

    Each field's section is capped at _MAX_FIELD_CONTEXT_WORDS so a single
    field with many large web chunks can't blow out the model's context window.
    """
    sections: list[str] = []
    for field_name, context in field_context.items():
        if isinstance(context, str):
            section_text = context
        else:
            section_text = _format_context(context)
        if not section_text:
            continue
        truncated, was_cut = _truncate_words(section_text, _MAX_FIELD_CONTEXT_WORDS)
        if was_cut:
            log.info(
                "synthesizer.context_truncated",
                field=field_name,
                original_words=len(section_text.split()),
                kept_words=_MAX_FIELD_CONTEXT_WORDS,
            )
        sections.append(f"[CONTEXT FOR {field_name.upper()}]\n{truncated}")
    return "\n\n".join(sections)


async def synthesize(
    latin_name: str,
    common_name: str | None,
    field_context: dict[str, list[RetrievedChunk] | str],
    llm: LLMProvider,
) -> SynthesizedFlower:
    """Generate all text fields for a flower from per-field context.

    field_context maps field name → context:
      - list[RetrievedChunk]: raw chunks for SIMPLE fields
      - str: pre-extracted facts for COMPLEX fields

    Fields absent from field_context are set to NOT_AVAILABLE.
    """
    if not field_context:
        return SynthesizedFlower()

    display_name = common_name or latin_name
    context_block = _format_per_field_context(field_context)
    available_fields = set(field_context.keys())

    all_text_fields = {
        "description", "fun_fact", "wiki_description",
        "habitat", "etymology", "cultural_info", "petal_color_hex",
    }
    unavailable = sorted(all_text_fields - available_fields)

    canonical_care = (
        '[{"icon":"sun.max.fill","label":"Full Sun"},'
        '{"icon":"moon.fill","label":"Full Shade"},'
        '{"icon":"cloud.sun.fill","label":"Part Shade"},'
        '{"icon":"drop","label":"Well Drained"},'
        '{"icon":"drop.fill","label":"Well Drained Soil"},'
        '{"icon":"drop.fill","label":"Moist Soil"},'
        '{"icon":"snowflake","label":"Fully Hardy"},'
        '{"icon":"snowflake","label":"Frost Hardy"},'
        '{"icon":"snowflake","label":"Half Hardy"},'
        '{"icon":"snowflake","label":"Tender"},'
        '{"icon":"drop.fill","label":"Water Plants"},'
        '{"icon":"drop.fill","label":"Wet Soil"}]'
    )

    na = NOT_AVAILABLE
    skip_list = unavailable if unavailable else "none"

    prompt = (
        f"You are a botanical data writer. Using ONLY the source material below, "
        f'generate JSON for the plant "{display_name}" ({latin_name}).\n\n'
        f"Each [CONTEXT FOR FIELD] section contains the relevant source material "
        f"for that specific field. Use ONLY the context in each section when "
        f"generating the corresponding field.\n\n"
        f"SOURCE MATERIAL:\n{context_block}\n\n"
        f"Generate a JSON object with these fields (use exactly these keys):\n"
        f'- "description": 2-3 sentence engaging description for a general audience '
        f'(skip if insufficient: "{na}")\n'
        f'- "fun_fact": one surprising or delightful fact (skip if insufficient: "{na}")\n'
        f'- "wiki_description": concise encyclopedic summary, 1-2 sentences '
        f'(skip if insufficient: "{na}")\n'
        f'- "habitat": native habitat and range description (skip if insufficient: "{na}")\n'
        f'- "etymology": meaning/origin of the latin name (skip if insufficient: "{na}")\n'
        f'- "cultural_info": historical or cultural significance '
        f'(skip if insufficient: "{na}")\n'
        f'- "petal_color_hex": dominant petal color as hex code like "#FF6B6B", '
        f"or null if unknown\n"
        f'- "care_info": array of objects chosen ONLY from the canonical list below '
        f"— pick all that apply based on PFAF data.\n"
        f"  Canonical values (use these exact icon/label strings):\n  {canonical_care}\n\n"
        f'Fields to set to "{na}": {skip_list}\n\n'
        f"Rules:\n"
        f"- Ground every claim in the provided sources. Do not invent facts.\n"
        f'- If a field cannot be answered from its context section, use "{na}"\n'
        f"- Return only valid JSON, no markdown, no explanation."
    )

    response = await llm.complete(
        prompt=prompt,
        system="You are a precise botanical content writer. Output only valid JSON.",
    )

    return _parse_response(response.text)


def _parse_response(response: str) -> SynthesizedFlower:
    """Extract and validate JSON from LLM response.

    On any failure (no JSON object, malformed JSON, schema mismatch) returns a
    default SynthesizedFlower with every field NOT_AVAILABLE — and logs why,
    so silent empty-output bugs are diagnosable.
    """
    text = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        log.warning(
            "synthesizer.parse_failed",
            reason="no_json_object",
            response_len=len(response),
            response_head=response[:500],
        )
        return SynthesizedFlower()

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        log.warning(
            "synthesizer.parse_failed",
            reason="json_decode_error",
            error=str(exc),
            response_len=len(response),
            response_head=response[:500],
        )
        return SynthesizedFlower()

    if not isinstance(data, dict):
        log.warning(
            "synthesizer.parse_failed",
            reason="non_object_json",
            response_len=len(response),
            response_head=response[:500],
        )
        return SynthesizedFlower()

    fields = SynthesizedFlower.model_fields
    filtered = {k: v for k, v in data.items() if k in fields}

    # Happy path: whole object validates
    try:
        return SynthesizedFlower(**filtered)
    except ValidationError as exc:
        log.warning(
            "synthesizer.partial_parse",
            error=str(exc)[:300],
            keys_found=list(filtered.keys()),
        )

    # Fallback: validate field-by-field so one bad field doesn't nuke the rest
    result = SynthesizedFlower()
    for field_name, value in filtered.items():
        try:
            validated = SynthesizedFlower.model_validate({field_name: value})
            setattr(result, field_name, getattr(validated, field_name))
        except ValidationError as exc:
            log.warning(
                "synthesizer.field_invalid",
                field=field_name,
                error=str(exc)[:200],
                value_preview=str(value)[:200],
            )
    return result
