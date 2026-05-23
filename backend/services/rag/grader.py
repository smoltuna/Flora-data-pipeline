"""Corrective RAG (CRAG) — grades retrieval quality before synthesis.

For each output field, grades whether retrieved chunks provide sufficient
evidence. Falls back gracefully: 'insufficient' → skip field (no hallucination).
"""
from __future__ import annotations

import re

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

RetrievalGrade = str  # 'sufficient' | 'partial' | 'insufficient'


async def grade_retrieval(
    field_name: str,
    latin_name: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider,
) -> tuple[RetrievalGrade, list[RetrievedChunk]]:
    """Grade all chunks' relevance to the target field in a single LLM call.

    Returns (grade, filtered_chunks) where grade determines synthesis strategy:
      sufficient  → standard RAG synthesis
      partial     → generate but mark confidence as low, flag for review
      insufficient → set field to 'Information not available.' — no hallucination
    """
    if not chunks:
        return "insufficient", []

    # Build a numbered list of chunk snippets for the LLM
    numbered = []
    for i, chunk in enumerate(chunks, 1):
        numbered.append(f"[{i}] {chunk.chunk_text[:500]}")
    chunks_block = "\n\n".join(numbered)

    response = await llm.complete(
        prompt=(
            f"Which of the following text chunks are relevant to generating "
            f"the '{field_name}' field for the plant {latin_name}?\n\n"
            f"{chunks_block}\n\n"
            f"Reply with ONLY the numbers of relevant chunks as a comma-separated list "
            f"(e.g. '1, 3'). If none are relevant, reply 'none'."
        ),
        system="You are a botanical data quality assessor. Be strict and concise.",
    )

    # Parse which chunk numbers the LLM marked as relevant
    text = response.strip().lower()
    if "none" in text and not any(c.isdigit() for c in text):
        relevant_indices: set[int] = set()
    else:
        relevant_indices = {
            int(m) for m in re.findall(r"\d+", text) if 1 <= int(m) <= len(chunks)
        }

    graded = [c for i, c in enumerate(chunks, 1) if i in relevant_indices]

    if len(graded) >= 2:
        return "sufficient", graded
    elif len(graded) == 1:
        return "partial", graded
    else:
        return "insufficient", []
