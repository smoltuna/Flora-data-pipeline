"""Hybrid BM25 + dense vector retrieval using Reciprocal Rank Fusion.

Per-flower retrieval strategy:
  - Dense: in-memory cosine similarity on pre-loaded pgvector embeddings
  - Sparse: tsvector BM25 via ts_rank_cd (PostgreSQL FTS, chunk_tsv column)
  - Fusion: Reciprocal Rank Fusion (RRF) merging both ranked lists per query
"""
from dataclasses import dataclass

import numpy as np
import structlog
from models import SourceEmbedding
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from services.embeddings.provider import EmbeddingProvider

log = structlog.get_logger()


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


async def _vector_search(
    candidates: list[RetrievedChunk],
    query: str,
    source_filter: list[str],
    top_k: int,
    embed_provider: EmbeddingProvider,
) -> list[RetrievedChunk]:
    """In-memory cosine similarity search over pre-loaded candidates."""
    filtered = [c for c in candidates if _source_matches(c.source, source_filter)]
    if not filtered:
        return []
    query_vec = await embed_provider.embed(query)
    scored = [
        (c, _cosine_similarity(query_vec, c.embedding))
        for c in filtered
        if c.embedding
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[tuple[RetrievedChunk, float]]:
    """Pure function. No DB, no async. Returns (chunk, fused_score) sorted desc.

    score(doc) = sum(1 / (rank_i + k)) across all lists containing the doc.
    k=60 is the standard value from the literature.
    """
    scores: dict[int, float] = {}
    chunk_map: dict[int, RetrievedChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rank + k)

    return sorted(
        [(chunk_map[cid], score) for cid, score in scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )


async def bm25_search(
    flower_id: int,
    query: str,
    source_filter: list[str],
    top_k: int,
    db: AsyncSession,
) -> list[RetrievedChunk]:
    """BM25 keyword search via PostgreSQL ts_rank_cd."""
    if not query.strip():
        return []
    try:
        async with db.begin_nested():
            result = await db.execute(
                text("""
                    SELECT id, chunk_text, metadata,
                           ts_rank_cd(chunk_tsv, plainto_tsquery('english', :query)) AS rank
                    FROM source_embeddings
                    WHERE flower_id = :flower_id
                      AND chunk_tsv @@ plainto_tsquery('english', :query)
                    ORDER BY rank DESC
                    LIMIT :limit
                """),
                {"flower_id": flower_id, "query": query, "limit": top_k * 3},
            )
            rows = result.mappings().all()
    except Exception as exc:
        log.warning("bm25_search.failed", error=str(exc), query=query[:80])
        return []

    chunks = []
    for row in rows:
        src = (row["metadata"] or {}).get("source", "unknown")
        if not _source_matches(src, source_filter):
            continue
        chunks.append(RetrievedChunk(
            chunk_id=row["id"],
            chunk_text=row["chunk_text"],
            source=src,
            rrf_score=float(row["rank"]),
            embedding=[],
        ))
    return chunks[:top_k]


async def hybrid_retrieve(
    flower_id: int,
    queries: list[str],
    hyde_doc: str | None,
    source_filter: list[str],
    top_k: int,
    db: AsyncSession,
    embed_provider: EmbeddingProvider,
) -> list[RetrievedChunk]:
    """Full hybrid retrieval: vector + BM25 per query, RRF fusion, top_k."""
    # Fetch all embeddings for this flower once — shared across all queries
    db_result = await db.execute(
        select(SourceEmbedding).where(SourceEmbedding.flower_id == flower_id)
    )
    all_rows = db_result.scalars().all()
    all_candidates = [
        RetrievedChunk(
            chunk_id=row.id,
            chunk_text=row.chunk_text,
            source=(row.metadata_ or {}).get("source", "unknown"),
            rrf_score=0.0,
            embedding=list(row.embedding) if row.embedding is not None else [],
        )
        for row in all_rows
    ]
    chunk_by_id: dict[int, RetrievedChunk] = {c.chunk_id: c for c in all_candidates}

    over_fetch = max(top_k * 2, 10)
    result_lists: list[list[RetrievedChunk]] = []

    for query in queries:
        vec_results = await _vector_search(
            all_candidates, query, source_filter, over_fetch, embed_provider
        )
        if vec_results:
            result_lists.append(vec_results)

        bm25_results = await bm25_search(flower_id, query, source_filter, over_fetch, db)
        # Fill embeddings for BM25 results from the in-memory pool
        for chunk in bm25_results:
            if not chunk.embedding and chunk.chunk_id in chunk_by_id:
                chunk.embedding = chunk_by_id[chunk.chunk_id].embedding
        if bm25_results:
            result_lists.append(bm25_results)

    if hyde_doc:
        hyde_results = await _vector_search(
            all_candidates, hyde_doc, source_filter, over_fetch, embed_provider
        )
        if hyde_results:
            result_lists.append(hyde_results)

    if not result_lists:
        return []

    fused = reciprocal_rank_fusion(result_lists)
    result_chunks = []
    for chunk, score in fused[:top_k]:
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
