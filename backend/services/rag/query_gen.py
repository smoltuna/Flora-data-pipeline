"""Multi-query and HyDE generation for COMPLEX field retrieval.

generate_field_queries — 3 diverse queries approaching the topic from different angles
generate_hyde_document — hypothetical ~100-word encyclopedia passage for HyDE
"""
from __future__ import annotations

from services.llm.provider import LLMProvider


async def generate_field_queries(
    latin_name: str,
    common_name: str | None,
    field_name: str,
    provider: LLMProvider,
) -> list[str]:
    """Generate 3 diverse queries for a COMPLEX field.

    Example for etymology of "Iris germanica" (common: Bearded Iris):
    → "origin and meaning of the name Iris germanica"
    → "etymology Iris flower Greek mythology goddess rainbow"
    → "why is bearded iris called Iris germanica naming history"
    """
    display = common_name or latin_name
    response = await provider.complete(
        prompt=(
            f"Generate exactly 3 diverse search queries to find information about the "
            f"'{field_name}' of the plant {latin_name} (common name: {display}).\n\n"
            f"Each query must approach the topic from a different angle. "
            f"Think: direct lookup, related keywords, and broader context.\n\n"
            f"Return ONLY the 3 queries, one per line, no numbering, no explanation."
        ),
        system="You are a search query generator. Output exactly 3 queries, one per line.",
    )

    lines = [line.strip() for line in response.text.strip().splitlines() if line.strip()]
    # Remove any accidental numbering like "1." or "-"
    cleaned = []
    for line in lines:
        line = line.lstrip("0123456789.-) ").strip()
        if line:
            cleaned.append(line)

    queries = cleaned[:3]
    if not queries:
        queries = [f"{latin_name} {field_name}"]
    return queries


async def generate_hyde_document(
    latin_name: str,
    common_name: str | None,
    field_name: str,
    provider: LLMProvider,
) -> str:
    """Generate a hypothetical ~100-word passage about this field for HyDE retrieval.

    Example for etymology of "Iris germanica":
    → "The name Iris germanica derives from Greek mythology. Iris was the goddess
       of the rainbow, and the genus was named by Carl Linnaeus in 1753 for the
       diverse colors found in its petals. The species epithet 'germanica' likely
       refers to Germany, though the species is not native there but was widely
       cultivated in German gardens..."

    The passage is embedded and used as a query vector — it matches real documents
    better than a short query phrase would.
    """
    display = common_name or latin_name
    response = await provider.complete(
        prompt=(
            f"Write an 80-120 word factual encyclopedia passage about the '{field_name}' "
            f"of {latin_name} ({display}). "
            f"Write in third person, present tense. Be specific and plausible. "
            f"This passage is used for document retrieval — the goal is to match relevant "
            f"source documents, not to be the final answer."
        ),
        system=(
            "You are a botanical encyclopedia writer. "
            "Write a concise, factual passage with no preamble."
        ),
    )
    return response.text.strip()
