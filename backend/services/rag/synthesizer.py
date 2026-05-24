"""LLM synthesis — constructs prompts from retrieved sources and generates structured output.

Output is validated against Pydantic v2 schemas. Uses source attribution
(according to PFAF... / Wikipedia states...) so the LLM grounds claims.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

NOT_AVAILABLE = "Information not available."


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
    source_labels = {
        "pfaf": "PFAF (Plants For A Future)",
        "wikipedia": "Wikipedia",
        "wikidata": "Wikidata",
        "gbif": "GBIF",
    }
    for source, texts in by_source.items():
        label = source_labels.get(source, source.upper())
        sections.append(f"[{label}]\n" + "\n---\n".join(texts))
    return "\n\n".join(sections)


async def synthesize(
    latin_name: str,
    common_name: str | None,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
    fields_to_skip: set[str] | None = None,
) -> SynthesizedFlower:
    """Generate all text fields for a flower from retrieved source chunks."""
    if not chunks:
        return SynthesizedFlower()

    skip = fields_to_skip or set()
    context = _format_context(chunks)
    display_name = common_name or latin_name

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
    skip_list = list(skip) if skip else "none"

    na = NOT_AVAILABLE
    prompt = (
        f"You are a botanical data writer. Using ONLY the source "
        f"material below, generate JSON for the plant "
        f'"{display_name}" ({latin_name}).\n\n'
        f"SOURCE MATERIAL:\n{context}\n\n"
        f"Generate a JSON object with these fields "
        f"(use exactly these keys):\n"
        f'- "description": 2-3 sentence engaging description '
        f'for a general audience (skip if insufficient: "{na}")\n'
        f'- "fun_fact": one surprising or delightful fact '
        f'(skip if insufficient: "{na}")\n'
        f'- "wiki_description": concise encyclopedic summary, '
        f'1-2 sentences (skip if insufficient: "{na}")\n'
        f'- "habitat": native habitat and range description '
        f'(skip if insufficient: "{na}")\n'
        f'- "etymology": meaning/origin of the latin name '
        f'(skip if insufficient: "{na}")\n'
        f'- "cultural_info": historical or cultural significance '
        f'(skip if insufficient: "{na}")\n'
        f'- "petal_color_hex": dominant petal color as hex code '
        f'like "#FF6B6B", or null if unknown\n'
        f'- "care_info": array of objects chosen ONLY from the '
        f"canonical list below — pick all that apply based on "
        f"PFAF data.\n"
        f"  Canonical values (use these exact icon/label "
        f"strings):\n  {canonical_care}\n\n"
        f'Fields to skip (set to "{na}"): {skip_list}\n\n'
        f"Rules:\n"
        f"- Ground every claim in the provided sources. "
        f"Do not invent facts.\n"
        f"- If a field cannot be answered from sources, "
        f'use "{na}"\n'
        f"- Return only valid JSON, no markdown, no explanation."
    )

    response = await llm.complete(
        prompt=prompt,
        system="You are a precise botanical content writer. "
        "Output only valid JSON.",
    )

    return _parse_response(response.text)


def _parse_response(response: str) -> SynthesizedFlower:
    """Extract and validate JSON from LLM response."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return SynthesizedFlower()

    try:
        data = json.loads(text[start:end])
        fields = SynthesizedFlower.model_fields
        filtered = {k: v for k, v in data.items() if k in fields}
        return SynthesizedFlower(**filtered)
    except (json.JSONDecodeError, Exception):
        return SynthesizedFlower()
