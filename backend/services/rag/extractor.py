"""Fact extraction — condenses noisy web chunks into clean summaries for synthesis."""
from __future__ import annotations

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

NOT_AVAILABLE = "Information not available."


async def extract_field_facts(
    field_name: str,
    latin_name: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
) -> str:
    """Extract 3-5 key facts from retrieved chunks relevant to the given field.

    Compresses noisy web text (~1500 tokens) into a clean ~200-token fact list.
    Only called for COMPLEX fields after CRAG grading confirms chunk relevance.
    Returns NOT_AVAILABLE if chunks are empty or extraction produces nothing.
    """
    if not chunks:
        return NOT_AVAILABLE

    numbered = "\n\n".join(
        f"[Source {i} — {chunk.source}]\n{chunk.chunk_text[:800]}"
        for i, chunk in enumerate(chunks, 1)
    )

    response = await llm.complete(
        prompt=(
            f"From the following sources about {latin_name}, extract 3-5 key facts "
            f"relevant to '{field_name}'. Be specific and concise. "
            f"Note which source each fact comes from.\n\n"
            f"{numbered}\n\n"
            f"Format your response as a bullet list, one fact per bullet."
        ),
        system="You are a botanical fact extractor. Extract only well-supported, specific facts.",
    )

    text = response.text.strip()
    return text if text else NOT_AVAILABLE
