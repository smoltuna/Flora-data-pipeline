"""Embedding service — chunks scraped content and stores vector embeddings.

Uses nomic-embed-text-v1.5 via Ollama (768-dim, Matryoshka-capable).

Source-aware chunking (Session 3):
  - pfaf / wikidata / gbif  → single chunk (structured, short)
  - wikipedia / web_ddg* → recursive splitting into ~chunk_size-token pieces
"""
from __future__ import annotations

from models import RawSource, SourceEmbedding
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.llm.provider import LLMProvider
from services.rag.chunker import RecursiveTextSplitter

# Sources that receive recursive chunking; all others are kept as one chunk.
_RECURSIVE_SOURCES = {"wikipedia", "web_ddg"}


def _should_chunk(source: str) -> bool:
    """Return True if this source should be split into multiple chunks."""
    return source in _RECURSIVE_SOURCES or source.startswith("web_ddg")


async def embed_and_store(
    flower_id: int,
    raw_source: RawSource,
    llm: LLMProvider,
    session: AsyncSession,
) -> list[SourceEmbedding]:
    """Embed a raw source (potentially as multiple chunks) and upsert into source_embeddings."""
    from config import settings

    text = _build_chunk_text(raw_source)
    if not text:
        return []

    if _should_chunk(raw_source.source):
        splitter = RecursiveTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split(text)
    else:
        chunks = [text]

    # Batch embed all chunks in one HTTP round-trip if Ollama, else sequential
    embeddings_vectors = await _batch_embed(chunks, llm)

    # Delete existing embeddings for this raw_source
    existing = await session.execute(
        select(SourceEmbedding).where(SourceEmbedding.raw_source_id == raw_source.id)
    )
    for row in existing.scalars().all():
        await session.delete(row)

    results: list[SourceEmbedding] = []
    for i, (chunk_text, vector) in enumerate(zip(chunks, embeddings_vectors)):
        emb = SourceEmbedding(
            raw_source_id=raw_source.id,
            flower_id=flower_id,
            chunk_text=chunk_text,
            embedding=vector,
            metadata_={"source": raw_source.source, "chunk_index": i},
        )
        session.add(emb)
        results.append(emb)

    await session.commit()
    for emb in results:
        await session.refresh(emb)
    return results


async def embed_all_sources(
    flower_id: int,
    llm: LLMProvider,
    session: AsyncSession,
) -> list[SourceEmbedding]:
    """Embed all raw sources for a flower that don't yet have embeddings."""
    result = await session.execute(
        select(RawSource).where(RawSource.flower_id == flower_id)
    )
    sources = result.scalars().all()

    all_embeddings: list[SourceEmbedding] = []
    for src in sources:
        if not src.raw_content and not src.parsed_content:
            continue
        embs = await embed_and_store(flower_id, src, llm, session)
        all_embeddings.extend(embs)
    return all_embeddings


async def _batch_embed(texts: list[str], llm: LLMProvider) -> list[list[float]]:
    """Embed a list of texts. Uses Ollama /api/embed batch endpoint when available,
    otherwise falls back to sequential embed() calls."""
    from services.llm.ollama import OllamaProvider

    if isinstance(llm, OllamaProvider) and len(texts) > 1:
        import httpx
        from config import settings

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{settings.ollama_base_url}/api/embed",
                    json={"model": settings.ollama_embed_model, "input": texts},
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]
        except Exception:
            pass  # fall through to sequential

    return [await llm.embed(t) for t in texts]


def _build_chunk_text(raw_source: RawSource) -> str:
    """Concatenate available text from a raw source into a single string."""
    parts: list[str] = []

    if raw_source.raw_content:
        # Trim to first 3000 chars — structured entries are short; web text gets chunked later
        parts.append(raw_source.raw_content[:3000])

    if raw_source.parsed_content:
        pc = raw_source.parsed_content
        if isinstance(pc, dict):
            for key, val in pc.items():
                if isinstance(val, str) and val.strip():
                    parts.append(f"{key}: {val}")
                elif isinstance(val, list):
                    flat = ", ".join(str(v) for v in val if v)
                    if flat:
                        parts.append(f"{key}: {flat}")

    return "\n".join(parts).strip()
