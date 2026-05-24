"""Hybrid BM25 + dense vector retrieval using Reciprocal Rank Fusion.

The hybrid_search SQL function (defined in the DB migration) combines:
  - Dense: HNSW cosine similarity on pgvector embeddings
  - Sparse: tsvector BM25 via ts_rank
  - Fusion: Reciprocal Rank Fusion (RRF) merging both ranked lists
"""
from dataclasses import dataclass

import numpy as np
from models import SourceEmbedding
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from services.embeddings.provider import EmbeddingProvider


@dataclass
class RetrievedChunk:
    chunk_id: int
    chunk_text: str
    source: str
    rrf_score: float
    embedding: list[float]


async def retrieve_for_flower(
    flower_id: int,
    session: AsyncSession,
) -> list[RetrievedChunk]:
    """Return all embedded chunks for a specific flower (deterministic filter)."""
    result = await session.execute(
        select(SourceEmbedding).where(SourceEmbedding.flower_id == flower_id)
    )
    rows = result.scalars().all()
    return [
        RetrievedChunk(
            chunk_id=row.id,
            chunk_text=row.chunk_text,
            source=(row.metadata_ or {}).get("source", "unknown"),
            rrf_score=1.0,
            embedding=list(row.embedding) if row.embedding is not None else [],
        )
        for row in rows
    ]


def _source_matches(source: str, source_filter: list[str]) -> bool:
    """Return True if source is included in source_filter.

    "all"     → always matches
    "web_ddg" → prefix match (handles web_ddg_0, web_ddg_1, …)
    others    → exact match
    """
    if "all" in source_filter:
        return True
    for f in source_filter:
        if source == f or source.startswith(f + "_"):
            return True
    return False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb)) / denom if denom > 0.0 else 0.0


async def retrieve_for_field(
    flower_id: int,
    field_name: str,
    queries: list[str],           # 1 for SIMPLE, 3 for COMPLEX
    hyde_doc: str | None,         # hypothetical document (if HyDE)
    source_filter: list[str],     # which sources to search; ["all"] = no filter
    top_k: int,
    db: AsyncSession,
    embed_provider: EmbeddingProvider,
) -> list[RetrievedChunk]:
    """Vector similarity search within flower_id + source_filter.

    For each query (and optional HyDE doc), embeds the query and scores all
    candidate chunks by cosine similarity. Merges results across queries by
    taking the maximum score per chunk, then returns top_k.

    Session 6 will add BM25 + RRF fusion to this function.
    """
    # Fetch all embeddings for this flower from DB
    result = await db.execute(
        select(SourceEmbedding).where(SourceEmbedding.flower_id == flower_id)
    )
    all_rows = result.scalars().all()

    # Filter by source
    if "all" not in source_filter:
        rows = [
            r for r in all_rows
            if _source_matches((r.metadata_ or {}).get("source", ""), source_filter)
        ]
    else:
        rows = list(all_rows)

    if not rows:
        return []

    # Build candidate pool
    candidates = [
        RetrievedChunk(
            chunk_id=row.id,
            chunk_text=row.chunk_text,
            source=(row.metadata_ or {}).get("source", "unknown"),
            rrf_score=0.0,
            embedding=list(row.embedding) if row.embedding is not None else [],
        )
        for row in rows
    ]

    # Collect all queries (including HyDE doc if provided)
    all_queries = list(queries)
    if hyde_doc:
        all_queries.append(hyde_doc)

    # For each query, compute cosine similarity against each candidate
    best_scores: dict[int, float] = {}  # chunk_id → max similarity across all queries
    for query_text in all_queries:
        query_vec = await embed_provider.embed(query_text)
        for chunk in candidates:
            if not chunk.embedding:
                continue
            sim = _cosine_similarity(query_vec, chunk.embedding)
            if sim > best_scores.get(chunk.chunk_id, -1.0):
                best_scores[chunk.chunk_id] = sim

    # Sort candidates by best score and return top_k
    scored = [
        (chunk, best_scores.get(chunk.chunk_id, 0.0))
        for chunk in candidates
        if chunk.chunk_id in best_scores
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    result_chunks = []
    for chunk, score in scored[:top_k]:
        chunk.rrf_score = score
        result_chunks.append(chunk)
    return result_chunks


async def hybrid_search(
    query_text: str,
    embed_provider: EmbeddingProvider,
    session: AsyncSession,
    match_count: int = 10,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Cross-flower hybrid search — used for gap detection and cross-flower queries."""
    query_embedding = await embed_provider.embed(query_text)

    # Call the SQL hybrid_search function (defined in migration)
    rows = await session.execute(
        text("""
            SELECT chunk_id, chunk_text, rrf_score
            FROM hybrid_search(:query_text, :query_embedding::vector, :match_count, :rrf_k)
        """),
        {
            "query_text": query_text,
            "query_embedding": str(query_embedding),
            "match_count": match_count,
            "rrf_k": rrf_k,
        },
    )

    chunks = []
    for row in rows:
        # Fetch embedding separately for gap detection
        emb_row = await session.execute(
            select(SourceEmbedding).where(SourceEmbedding.id == row.chunk_id)
        )
        emb_obj = emb_row.scalar_one_or_none()
        chunks.append(RetrievedChunk(
            chunk_id=row.chunk_id,
            chunk_text=row.chunk_text,
            source="unknown",
            rrf_score=row.rrf_score,
            embedding=list(emb_obj.embedding) if emb_obj and emb_obj.embedding is not None else [],
        ))
    return chunks
