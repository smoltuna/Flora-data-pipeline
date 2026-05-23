"""Sequential pipeline orchestration — runs full enrichment for one flower.

Pipeline stages (in order):
  1. Scrape all sources (PFAF, Wikipedia, Wikidata, GBIF)
  2. Embed all sources → store in source_embeddings
  3. Retrieve all chunks for this flower
  4. Semantic deduplication
  5. Adaptive RAG routing (full / sparse / minimal based on source coverage)
  6. Per-field Corrective RAG grading
  7. LLM synthesis
  8. Self-RAG verification
  9. Persist enriched flower + confidence scores
 10. Translate into all supported languages
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from datetime import date

import mlflow
import structlog
from config import settings
from models import Flower, RawSource
from routers.scrape import _do_scrape
from services.llm.provider import LLMProvider, get_provider
from services.rag.deduplicator import deduplicate_chunks
from services.rag.embedder import embed_and_store
from services.rag.grader import grade_retrieval
from services.rag.retriever import RetrievedChunk, retrieve_for_flower
from services.rag.synthesizer import NOT_AVAILABLE, SynthesizedFlower, synthesize
from services.rag.verifier import verify_all_fields
from services.tracing import PipelineTracer
from services.translation.translator import translate_flower
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger()

TEXT_FIELDS = [
    "description", "fun_fact", "wiki_description",
    "habitat", "etymology", "cultural_info",
]


def _mlflow_context(latin_name: str, flower_id: int):
    """Return an MLflow run context manager, or a no-op if server is unavailable."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("flora-enrichment")
        return mlflow.start_run(run_name=f"{latin_name}-{flower_id}")
    except Exception:
        return nullcontext()


def _log_mlflow_metrics(
    confidence_scores: dict, n_chunks: int,
    n_deduped: int, elapsed: float,
) -> None:
    """Log metrics to the active MLflow run; silently skip if no run is active."""
    try:
        flat: dict[str, float] = {
            "pipeline_duration_s": elapsed,
            "chunks_retrieved": float(n_chunks),
            "chunks_after_dedup": float(n_deduped),
        }
        for field, scores in confidence_scores.items():
            flat[f"confidence_llm_{field}"] = scores.get("llm_score", 0.0)
        mlflow.log_metrics(flat)
    except Exception:
        pass


async def run_pipeline(
    flower_id: int,
    db: AsyncSession,
    feature_date: date | None = None,
    _batch_traces: list[dict] | None = None,
    _images_fn: Callable[[], Awaitable[None]] | None = None,
) -> Flower:
    """Run the full enrichment pipeline for a single flower. Returns the updated Flower."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise ValueError(f"Flower {flower_id} not found")

    tracer = PipelineTracer(flower_id, flower.latin_name)
    log.info("pipeline.start", flower_id=flower_id, latin_name=flower.latin_name)
    start_time = time.perf_counter()

    # Initialise variables used in finally so they're always defined.
    chunks: list[RetrievedChunk] = []
    deduped: list[RetrievedChunk] = []
    confidence_scores: dict = {}

    with _mlflow_context(flower.latin_name, flower_id):
        try:
            mlflow.set_tags({
                "latin_name": flower.latin_name,
                "flower_id": str(flower_id),
                "llm_provider": settings.llm_provider,
            })
        except Exception:
            pass

        try:
            # ── Stage 1: Scrape ─────────────────────────────────────────────
            flower.status = "scraping"
            await db.commit()
            with tracer.step("scrape") as m:
                scrape_result = await _do_scrape(flower_id, flower.latin_name, db)
                m.api_calls = (
                    len(scrape_result.sources_scraped) + len(scrape_result.sources_failed)
                )
            log.info(
                "pipeline.scraped",
                sources=scrape_result.sources_scraped,
                failed=scrape_result.sources_failed,
            )

            # ── Stage 2: Embed ──────────────────────────────────────────────
            flower.status = "embedding"
            await db.commit()
            llm: LLMProvider = get_provider()
            await db.refresh(flower)

            pfaf_raw_care: dict | None = None
            embed_success = 0
            embed_failed = 0
            with tracer.step("embed") as m:
                sources_result = await db.execute(
                    select(RawSource).where(RawSource.flower_id == flower_id)
                )
                raw_sources = sources_result.scalars().all()
                for src in raw_sources:
                    if src.source == "pfaf" and src.parsed_content:
                        care_info = src.parsed_content.get("care_info")
                        if isinstance(care_info, dict) and care_info:
                            # Keep original PFAF labels/values as the canonical care info.
                            pfaf_raw_care = care_info
                            break

                for src in raw_sources:
                    if src.raw_content or src.parsed_content:
                        try:
                            await embed_and_store(flower_id, src, llm, db)
                            embed_success += 1
                        except Exception as e:
                            embed_failed += 1
                            log.warning("pipeline.embed_failed", source=src.source, error=str(e))
                m.api_calls = embed_success + embed_failed
                m.chunks_out = embed_success

            if raw_sources and embed_success == 0:
                flower.status = "failed"
                await db.commit()
                raise RuntimeError(
                    "No embeddings were created. Ensure Ollama is running and "
                    "OLLAMA_EMBED_MODEL is available."
                )

            if embed_failed:
                log.info("pipeline.embed_summary", succeeded=embed_success, failed=embed_failed)

            # ── Stage 3: Retrieve ───────────────────────────────────────────
            with tracer.step("retrieve") as m:
                chunks = await retrieve_for_flower(flower_id, db)
                m.chunks_out = len(chunks)
            log.info("pipeline.retrieved", n_chunks=len(chunks))

            if not chunks:
                flower.status = "failed"
                await db.commit()
                raise RuntimeError(
                    "No retrieved chunks found after embedding. Pipeline stopped to avoid "
                    "saving low-confidence empty enrichment output."
                )

            # ── Stage 4: Semantic deduplication ────────────────────────────
            with tracer.step("dedup") as m:
                deduped = deduplicate_chunks(chunks)
                m.chunks_in = len(chunks)
                m.chunks_out = len(deduped)
            log.info("pipeline.deduped", before=len(chunks), after=len(deduped))

            # ── Stage 5: Synthesize ─────────────────────────────────────────
            sources_present = {c.source for c in deduped}
            with tracer.step("synthesize") as m:
                synthesis_result = await _adaptive_synthesize(
                    flower, deduped, sources_present, llm
                )
                m.chunks_in = len(deduped)

            # ── Stage 6: CRAG grading ───────────────────────────────────────
            _source_order = {"wikipedia": 0, "wikidata": 1, "gbif": 2, "pfaf": 3}
            verification_chunks = sorted(
                deduped, key=lambda c: _source_order.get(c.source, 4)
            )
            source_text = "\n\n".join(c.chunk_text for c in verification_chunks)
            generated_fields = {
                f: getattr(synthesis_result, f, NOT_AVAILABLE)
                for f in TEXT_FIELDS
                if getattr(synthesis_result, f, NOT_AVAILABLE) != NOT_AVAILABLE
            }

            with tracer.step("grade") as m:
                m.chunks_in = len(deduped)
                for field_name in list(generated_fields.keys()):
                    grade, _ = await grade_retrieval(
                        field_name, flower.latin_name, deduped, llm
                    )
                    if grade == "insufficient":
                        generated_fields[field_name] = NOT_AVAILABLE
                    elif grade == "partial":
                        log.info("pipeline.crag_partial", field=field_name)

            # ── Stage 7: Self-RAG verification ─────────────────────────────
            fields_to_verify = {
                f: v for f, v in generated_fields.items() if v != NOT_AVAILABLE
            }
            with tracer.step("verify") as m:
                verification_results = await verify_all_fields(
                    fields_to_verify, source_text, llm
                )

            confidence_scores = {
                field: {"llm_score": res.confidence}
                for field, res in verification_results.items()
            }

            # ── Persist ─────────────────────────────────────────────────────
            flower.status = "enriched"
            flower.description = synthesis_result.description
            flower.fun_fact = synthesis_result.fun_fact
            flower.wiki_description = synthesis_result.wiki_description
            flower.habitat = synthesis_result.habitat
            flower.etymology = synthesis_result.etymology
            flower.cultural_info = synthesis_result.cultural_info
            flower.petal_color_hex = synthesis_result.petal_color_hex
            if pfaf_raw_care:
                flower.care_info = pfaf_raw_care  # type: ignore[assignment]
            elif synthesis_result.care_info:
                flower.care_info = synthesis_result.care_info  # type: ignore[assignment]
            flower.confidence_scores = confidence_scores

            if not flower.feature_month:
                d = feature_date or date(2026, 5, 1)
                flower.feature_year = d.year
                flower.feature_month = d.month
                flower.feature_day = d.day

            await db.commit()
            await db.refresh(flower)

            # ── Stage 8: Translate + Images (concurrent) ────────────────────
            log.info("pipeline.translating", flower_id=flower_id)

            async def _translate_traced() -> None:
                with tracer.step("translate") as m:
                    try:
                        await translate_flower(flower_id, db)
                        log.info("pipeline.translated", flower_id=flower_id)
                    except Exception as e:
                        log.warning("pipeline.translate_failed", flower_id=flower_id, error=str(e))
                        m.errors.append(str(e))

            if _images_fn is not None:
                async def _images_traced() -> None:
                    with tracer.step("images") as m:
                        await _images_fn()
                        m.api_calls = 3  # wikimedia + fal main + fal lock

                await asyncio.gather(_translate_traced(), _images_traced())
            else:
                await _translate_traced()

        finally:
            elapsed = time.perf_counter() - start_time
            if chunks:
                _log_mlflow_metrics(confidence_scores, len(chunks), len(deduped), elapsed)
            tracer.log_summary()
            if _batch_traces is not None:
                _batch_traces.append(tracer.summary())
            log.info(
                "pipeline.complete",
                flower_id=flower_id,
                status=flower.status,
                elapsed_s=round(elapsed, 2),
            )

        return flower


async def _adaptive_synthesize(
    flower: Flower,
    chunks: list[RetrievedChunk],
    sources_present: set[str],
    llm: LLMProvider,
) -> SynthesizedFlower:
    """Route synthesis based on source coverage."""
    if "pfaf" in sources_present and "wikipedia" in sources_present:
        return await synthesize(flower.latin_name, flower.common_name, chunks, llm)
    elif "wikidata" in sources_present or "gbif" in sources_present:
        return await synthesize(
            flower.latin_name, flower.common_name, chunks, llm,
            fields_to_skip={"fun_fact", "cultural_info"},
        )
    else:
        return await synthesize(
            flower.latin_name, flower.common_name, chunks, llm,
            fields_to_skip={"fun_fact", "cultural_info", "etymology"},
        )
