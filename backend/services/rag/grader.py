"""Corrective RAG (CRAG) — grades retrieval quality before synthesis.

For each output field, grades whether retrieved chunks provide sufficient
evidence. Falls back gracefully: 'insufficient' → skip field (no hallucination).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import structlog

from services.llm.provider import LLMProvider
from services.rag.retriever import RetrievedChunk

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from services.embeddings.provider import EmbeddingProvider
    from services.rag.router import FieldConfig

log = structlog.get_logger()

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
    text = response.text.strip().lower()
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


async def grade_and_correct(
    flower_id: int,
    latin_name: str,
    common_name: str | None,
    field_name: str,
    chunks: list[RetrievedChunk],
    field_config: "FieldConfig",
    db: "AsyncSession",
    llm: LLMProvider,
    embed: "EmbeddingProvider",
) -> tuple[list[RetrievedChunk], str]:
    """CRAG with one round of web search correction for COMPLEX fields.

    Grades retrieved chunks. If grade is 'insufficient' and the field is COMPLEX,
    runs a targeted web search, embeds results in-memory, and re-grades the combined
    chunk set. Max 1 correction round to bound LLM + embed costs.
    """
    from services.rag.router import FieldDifficulty

    grade, graded = await grade_retrieval(field_name, latin_name, chunks, llm)

    if grade != "insufficient" or field_config.difficulty != FieldDifficulty.COMPLEX:
        return graded, grade

    log.info(
        "grader.crag_correction",
        field=field_name,
        latin_name=latin_name,
        original_chunks=len(chunks),
    )

    # Targeted web search for the specific knowledge gap
    from services.scraper.web_search import search_web
    targeted_query = f"{latin_name} {field_name.replace('_', ' ')}"
    web_results = await search_web(
        latin_name, common_name, max_results=3, query=targeted_query
    )
    if not web_results:
        return graded, grade

    # Chunk + embed web results in memory
    from services.rag.chunker import RecursiveTextSplitter

    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks: list[str] = []
    for result in web_results:
        text_chunks.extend(splitter.split(result.content))

    if not text_chunks:
        return graded, grade

    vectors = await embed.embed_batch(text_chunks)
    query_vec = await embed.embed(targeted_query)
    qa = np.array(query_vec, dtype=np.float32)

    # Score new chunks by cosine similarity to the targeted query
    scored: list[tuple[int, float]] = []
    for i, vec in enumerate(vectors):
        vb = np.array(vec, dtype=np.float32)
        denom = float(np.linalg.norm(qa) * np.linalg.norm(vb))
        sim = float(np.dot(qa, vb)) / denom if denom > 0.0 else 0.0
        scored.append((i, sim))
    scored.sort(key=lambda x: x[1], reverse=True)

    new_chunks = [
        RetrievedChunk(
            chunk_id=-(rank + 1),  # negative IDs for ephemeral correction chunks
            chunk_text=text_chunks[idx],
            source="web_ddg_correction",
            rrf_score=sim,
            embedding=vectors[idx],
        )
        for rank, (idx, sim) in enumerate(scored[:field_config.top_k])
    ]

    combined = new_chunks + chunks
    grade2, graded2 = await grade_retrieval(field_name, latin_name, combined, llm)
    log.info(
        "grader.crag_corrected",
        field=field_name,
        grade_before="insufficient",
        grade_after=grade2,
        new_chunks=len(new_chunks),
    )
    return graded2, grade2
