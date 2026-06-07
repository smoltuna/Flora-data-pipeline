"""Pipeline orchestration — runs full enrichment for one flower.

Pipeline stages (in order):
  1. Scrape all sources (PFAF, Wikipedia, Wikidata, GBIF, web)
  2. Embed all sources → store in source_embeddings
  3. Per-field hybrid retrieval (BM25 + vector + RRF, multi-query + HyDE for complex)
  4. CRAG grade + targeted web search correction
  5. Semantic deduplication of graded chunks
  6. Fact extraction (COMPLEX fields only)
  7. LLM synthesis with per-field context
  8. Self-RAG verification with source confidence weighting
  9. LLM-as-Judge evaluation (5 criteria per field)
 10. Persist enriched flower + confidence/judge scores
 11. Translate into 6 languages, concurrent with image processing
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
from services.embeddings.provider import EmbeddingProvider, get_embedding_provider
from services.llm.provider import LLMProvider, get_provider
from services.observability import (
    get_tracer,
    record_crag_grade,
    record_field_confidence,
    record_pipeline_completion,
    record_pipeline_failure,
    step_span,
)
from services.rag.deduplicator import deduplicate_chunks
from services.rag.embedder import embed_and_store
from services.rag.extractor import extract_field_facts
from services.rag.fact_checker import fact_check_fields
from services.rag.grader import grade_and_correct
from services.rag.judge import judge_flower
from services.rag.query_gen import generate_field_queries, generate_hyde_document
from services.rag.retriever import RetrievedChunk, hybrid_retrieve
from services.rag.router import FIELD_CONFIG, FieldDifficulty
from services.rag.synthesizer import NOT_AVAILABLE, synthesize
from services.rag.verifier import verify_all_fields
from services.scraper.orchestrator import scrape_all_sources
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
    _images_fn: Callable[[], Awaitable[None]] | None = None,
) -> Flower:
    """Run the full enrichment pipeline for a single flower. Returns the updated Flower."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise ValueError(f"Flower {flower_id} not found")

    log.info("pipeline.start", flower_id=flower_id, latin_name=flower.latin_name)
    start_time = time.perf_counter()

    # Initialise variables used in finally so they're always defined.
    chunks: list[RetrievedChunk] = []
    deduped: list[RetrievedChunk] = []
    confidence_scores: dict = {}

    otel_tracer = get_tracer()
    with otel_tracer.start_as_current_span(
        "flower",
        attributes={"latin_name": flower.latin_name, "flower_id": flower_id},
    ), _mlflow_context(flower.latin_name, flower_id):
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
            with step_span("scrape") as span:
                scrape_result = await scrape_all_sources(flower_id, flower.latin_name, db)
                span.set_attribute(
                    "api_calls",
                    len(scrape_result.sources_scraped) + len(scrape_result.sources_failed),
                )
            log.info(
                "pipeline.scraped",
                sources=scrape_result.sources_scraped,
                failed=scrape_result.sources_failed,
            )

            # ── Stage 2: Embed ──────────────────────────────────────────────
            flower.status = "embedding"
            await db.commit()
            llm: LLMProvider = get_provider(step="synth")
            embed_provider: EmbeddingProvider = get_embedding_provider()
            grade_llm: LLMProvider = get_provider(step="grade")
            await db.refresh(flower)

            pfaf_raw_care: dict | None = None
            embed_success = 0
            embed_failed = 0
            with step_span("embed") as span:
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
                            await embed_and_store(flower_id, src, embed_provider, db)
                            embed_success += 1
                        except Exception as e:
                            embed_failed += 1
                            log.warning("pipeline.embed_failed", source=src.source, error=str(e))
                span.set_attribute("api_calls", embed_success + embed_failed)
                span.set_attribute("chunks_out", embed_success)

            if raw_sources and embed_success == 0:
                flower.status = "failed"
                await db.commit()
                raise RuntimeError(
                    "No embeddings were created. Ensure Ollama is running and "
                    "OLLAMA_EMBED_MODEL is available."
                )

            if embed_failed:
                log.info("pipeline.embed_summary", succeeded=embed_success, failed=embed_failed)

            # ── Stage 3: Field-specific retrieval ──────────────────────────
            query_gen_llm: LLMProvider = get_provider(step="query_gen")
            per_field_chunks: dict[str, list[RetrievedChunk]] = {}

            with step_span("retrieve") as span:
                for field_name, config in FIELD_CONFIG.items():
                    if config.difficulty == FieldDifficulty.NONE:
                        continue

                    if config.difficulty == FieldDifficulty.SIMPLE:
                        queries = [f"{flower.latin_name} {field_name}"]
                        hyde_doc = None
                    else:  # COMPLEX
                        queries = await generate_field_queries(
                            flower.latin_name, flower.common_name, field_name, query_gen_llm
                        )
                        hyde_doc = None
                        if config.use_hyde:
                            hyde_doc = await generate_hyde_document(
                                flower.latin_name, flower.common_name, field_name, query_gen_llm
                            )

                    field_chunks = await hybrid_retrieve(
                        flower_id=flower_id,
                        queries=queries,
                        hyde_doc=hyde_doc,
                        source_filter=config.sources,
                        top_k=config.top_k,
                        db=db,
                        embed_provider=embed_provider,
                    )
                    per_field_chunks[field_name] = field_chunks
                    log.debug(
                        "pipeline.field_retrieved",
                        field=field_name,
                        difficulty=config.difficulty.value,
                        n_chunks=len(field_chunks),
                    )

                # Union per-field chunks into a flat list for total count
                seen_ids: set[int] = set()
                chunks = []
                for field_chunks_list in per_field_chunks.values():
                    for chunk in field_chunks_list:
                        if chunk.chunk_id not in seen_ids:
                            seen_ids.add(chunk.chunk_id)
                            chunks.append(chunk)
                span.set_attribute("chunks_out", len(chunks))

            log.info("pipeline.retrieved", n_chunks=len(chunks), n_fields=len(per_field_chunks))

            if not chunks:
                flower.status = "failed"
                await db.commit()
                raise RuntimeError(
                    "No retrieved chunks found after embedding. Pipeline stopped to avoid "
                    "saving low-confidence empty enrichment output."
                )

            # ── Stage 4: CRAG grade + correct ──────────────────────────────
            graded_per_field: dict[str, list[RetrievedChunk]] = {}
            with step_span("grade") as span:
                span.set_attribute("chunks_in", len(chunks))
                for field_name, field_chunks_list in per_field_chunks.items():
                    field_cfg = FIELD_CONFIG[field_name]
                    final_chunks, grade = await grade_and_correct(
                        flower_id, flower.latin_name, flower.common_name, field_name,
                        field_chunks_list, field_cfg, db, grade_llm, embed_provider,
                    )
                    graded_per_field[field_name] = final_chunks  # [] if insufficient
                    record_crag_grade(field_name, grade)
                    if grade == "partial":
                        log.info("pipeline.crag_partial", field=field_name)
                    elif grade == "insufficient":
                        log.debug("pipeline.crag_insufficient", field=field_name)

            # ── Stage 4b: Semantic deduplication (over graded chunks) ───────
            seen_graded: set[int] = set()
            all_graded_chunks: list[RetrievedChunk] = []
            for fc in graded_per_field.values():
                for chunk in fc:
                    # Skip ephemeral correction chunks (negative IDs) from dedup
                    if chunk.chunk_id >= 0 and chunk.chunk_id not in seen_graded:
                        seen_graded.add(chunk.chunk_id)
                        all_graded_chunks.append(chunk)

            with step_span("dedup") as span:
                deduped = deduplicate_chunks(all_graded_chunks)
                span.set_attribute("chunks_in", len(all_graded_chunks))
                span.set_attribute("chunks_out", len(deduped))
            log.info("pipeline.deduped", before=len(all_graded_chunks), after=len(deduped))

            # ── Stage 5: Extract facts for COMPLEX fields ───────────────────
            field_context: dict[str, list[RetrievedChunk] | str] = {}
            with step_span("extract") as span:
                span.set_attribute("chunks_in", len(all_graded_chunks))
                for field_name, field_chunks_list in graded_per_field.items():
                    if not field_chunks_list:
                        continue  # insufficient → absent from context → NOT_AVAILABLE
                    field_cfg = FIELD_CONFIG[field_name]
                    if field_cfg.difficulty == FieldDifficulty.COMPLEX:
                        facts = await extract_field_facts(
                            field_name, flower.latin_name, field_chunks_list, llm
                        )
                        if facts and facts != NOT_AVAILABLE:
                            field_context[field_name] = facts
                    else:
                        field_context[field_name] = field_chunks_list

            # ── Stage 6: Synthesize ─────────────────────────────────────────
            with step_span("synthesize") as span:
                synthesis_result = await synthesize(
                    flower.latin_name, flower.common_name, field_context, llm
                )
                span.set_attribute("chunks_in", len(all_graded_chunks))

            # ── Stage 6b: Post-synthesis web fact-check for COMPLEX fields ─
            # Catches hallucinations the synthesizer makes over correct chunks
            # (e.g. Tulipa etymology wrote "Joannes Andreae Gesner" when Wikipedia
            # says Conrad Gesner). One regen per field, gated by config flag.
            fact_check_llm: LLMProvider = get_provider(step="fact_check")
            with step_span("fact_check") as span:
                synth_for_check = {
                    f: getattr(synthesis_result, f, NOT_AVAILABLE)
                    for f in ("etymology", "cultural_info", "fun_fact")
                }
                checks = await fact_check_fields(
                    flower.latin_name, flower.common_name,
                    synth_for_check, fact_check_llm,
                )
                n_regen = 0
                for field_name, result in checks.items():
                    if result.corrected:
                        setattr(synthesis_result, field_name, result.corrected)
                        n_regen += 1
                span.set_attribute("regenerated", n_regen)
            if checks:
                log.info("pipeline.fact_checked", n_checked=len(checks), n_regen=n_regen)

            # ── Stage 7: Self-RAG verification ─────────────────────────────
            _source_order = {"wikipedia": 0, "wikidata": 1, "gbif": 2, "pfaf": 3}
            verification_chunks = sorted(
                deduped, key=lambda c: _source_order.get(c.source, 4)
            )
            source_text = "\n\n".join(c.chunk_text for c in verification_chunks)
            fields_to_verify = {
                f: getattr(synthesis_result, f, NOT_AVAILABLE)
                for f in TEXT_FIELDS
                if getattr(synthesis_result, f, NOT_AVAILABLE) != NOT_AVAILABLE
            }
            with step_span("verify"):
                verification_results = await verify_all_fields(
                    fields_to_verify, source_text, llm,
                    field_chunks={f: graded_per_field.get(f, []) for f in fields_to_verify},
                )

            confidence_scores = {
                field: {"llm_score": res.confidence}
                for field, res in verification_results.items()
            }
            for field, res in verification_results.items():
                record_field_confidence(field, res.confidence)

            # ── Stage 7b: LLM-as-Judge ─────────────────────────────────────
            judge_llm: LLMProvider = get_provider(step="judge")
            with step_span("judge"):
                judge_scores = await judge_flower(
                    flower.latin_name,
                    fields_to_verify,
                    {f: graded_per_field.get(f, []) for f in fields_to_verify},
                    judge_llm,
                )
            for field, scores in judge_scores.items():
                confidence_scores.setdefault(field, {})["judge"] = scores

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
                with step_span("translate") as span:
                    try:
                        await translate_flower(flower_id, db)
                        log.info("pipeline.translated", flower_id=flower_id)
                    except Exception as e:
                        log.warning("pipeline.translate_failed", flower_id=flower_id, error=str(e))
                        span.record_exception(e)

            if _images_fn is not None:
                async def _images_traced() -> None:
                    with step_span("images") as span:
                        await _images_fn()
                        span.set_attribute("api_calls", 3)  # wikimedia + fal main + fal lock

                await asyncio.gather(_translate_traced(), _images_traced())
            else:
                await _translate_traced()

        finally:
            elapsed = time.perf_counter() - start_time
            if chunks:
                _log_mlflow_metrics(confidence_scores, len(chunks), len(deduped), elapsed)
            if flower.status == "enriched":
                record_pipeline_completion(elapsed)
            elif flower.status == "failed":
                record_pipeline_failure()
            log.info(
                "pipeline.complete",
                flower_id=flower_id,
                status=flower.status,
                elapsed_s=round(elapsed, 2),
            )

        return flower


