"""RAG enrichment endpoints — trigger pipeline stages for a flower."""
from __future__ import annotations

from database import get_db
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models import Flower
from pydantic import BaseModel
from services.embeddings.provider import get_embedding_provider
from services.rag.deduplicator import deduplicate_chunks
from services.rag.embedder import embed_all_sources
from services.rag.retriever import retrieve_for_flower
from sqlalchemy.ext.asyncio import AsyncSession
from tasks.pipeline import run_pipeline

router = APIRouter()


class EnrichResult(BaseModel):
    flower_id: int
    latin_name: str
    status: str
    confidence_scores: dict | None = None


class EmbedResult(BaseModel):
    flower_id: int
    embeddings_created: int


class ChunkOut(BaseModel):
    chunk_id: int
    chunk_text: str
    source: str
    rrf_score: float


@router.post("/{flower_id}", response_model=EnrichResult)
async def enrich_flower(
    flower_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> EnrichResult:
    """Run full RAG pipeline in background."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    background_tasks.add_task(_run_pipeline_bg, flower_id)
    return EnrichResult(flower_id=flower_id, latin_name=flower.latin_name, status="queued")


@router.post("/{flower_id}/sync", response_model=EnrichResult)
async def enrich_flower_sync(flower_id: int, db: AsyncSession = Depends(get_db)) -> EnrichResult:
    """Synchronous enrichment — waits for pipeline completion."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    updated = await run_pipeline(flower_id, db)
    return EnrichResult(
        flower_id=flower_id,
        latin_name=updated.latin_name,
        status=updated.status,
        confidence_scores=updated.confidence_scores,
    )


@router.post("/{flower_id}/embed", response_model=EmbedResult)
async def embed_flower(flower_id: int, db: AsyncSession = Depends(get_db)) -> EmbedResult:
    """Embed all raw sources for a flower (without running full pipeline)."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    embed_provider = get_embedding_provider()
    embeddings = await embed_all_sources(flower_id, embed_provider, db)
    return EmbedResult(flower_id=flower_id, embeddings_created=len(embeddings))


@router.get("/{flower_id}/chunks", response_model=list[ChunkOut])
async def get_chunks(
    flower_id: int,
    deduplicated: bool = False,
    db: AsyncSession = Depends(get_db),
) -> list[ChunkOut]:
    """Return retrieved chunks for a flower (optionally deduplicated)."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    chunks = await retrieve_for_flower(flower_id, db)
    if deduplicated:
        chunks = deduplicate_chunks(chunks)

    return [
        ChunkOut(
            chunk_id=c.chunk_id,
            chunk_text=c.chunk_text,
            source=c.source,
            rrf_score=c.rrf_score,
        )
        for c in chunks
    ]


async def _run_pipeline_bg(flower_id: int) -> None:
    from database import async_session_factory
    async with async_session_factory() as session:
        await run_pipeline(flower_id, session)
