"""Scraping trigger endpoints — kick off per-source scraping for a flower."""
from __future__ import annotations

import asyncio

from database import get_db
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models import Flower, RawSource
from pydantic import BaseModel
from services.scraper.gbif import fetch_gbif
from services.scraper.pfaf import scrape_pfaf
from services.scraper.wikidata import fetch_wikidata
from services.scraper.wikipedia import fetch_wikipedia
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


class ScrapeResult(BaseModel):
    flower_id: int
    latin_name: str
    sources_scraped: list[str]
    sources_failed: list[str]


@router.post("/{flower_id}", response_model=ScrapeResult)
async def scrape_flower(
    flower_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ScrapeResult:
    """Trigger scraping for all sources for a flower. Runs in background."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    flower.status = "scraping"
    await db.commit()

    background_tasks.add_task(_run_scrape, flower_id, flower.latin_name)
    return ScrapeResult(
        flower_id=flower_id,
        latin_name=flower.latin_name,
        sources_scraped=[],
        sources_failed=[],
    )


@router.post("/{flower_id}/sync", response_model=ScrapeResult)
async def scrape_flower_sync(flower_id: int, db: AsyncSession = Depends(get_db)) -> ScrapeResult:
    """Synchronous scraping (waits for completion). Useful for testing."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    return await _do_scrape(flower_id, flower.latin_name, db)


async def _run_scrape(flower_id: int, latin_name: str) -> None:
    """Background task — creates its own DB session."""
    from database import async_session_factory
    async with async_session_factory() as session:
        await _do_scrape(flower_id, latin_name, session)


async def _do_scrape(flower_id: int, latin_name: str, db: AsyncSession) -> ScrapeResult:
    scraped: list[str] = []
    failed: list[str] = []

    # Run all 4 scrapers concurrently. PFAF has a 2-sec courtesy delay internally,
    # but it now overlaps with the other 3 sources, cutting total scrape time to ~2-3s.
    pfaf_res, wiki_res, wd_res, gbif_res = await asyncio.gather(
        scrape_pfaf(latin_name),
        fetch_wikipedia(latin_name),
        fetch_wikidata(latin_name),
        fetch_gbif(latin_name),
        return_exceptions=True,
    )

    # Process results sequentially — DB writes on a shared session must not be concurrent.

    # PFAF
    if isinstance(pfaf_res, BaseException) or pfaf_res is None:
        failed.append("pfaf")
    else:
        try:
            pfaf_data = pfaf_res
            await _upsert_source(db, flower_id, "pfaf",
                raw_content=pfaf_data.raw_text,
                parsed={
                    "common_name": pfaf_data.common_name,
                    "edibility_rating": pfaf_data.edibility_rating,
                    "medicinal_rating": pfaf_data.medicinal_rating,
                    "other_uses_rating": pfaf_data.other_uses_rating,
                    "weed_potential": pfaf_data.weed_potential,
                    "habitat": pfaf_data.habitat,
                    "care_info": pfaf_data.care_info,
                })
            flower = await db.get(Flower, flower_id)
            if flower:
                if pfaf_data.common_name and not flower.common_name:
                    flower.common_name = pfaf_data.common_name
                flower.edibility_rating = pfaf_data.edibility_rating
                flower.medicinal_rating = pfaf_data.medicinal_rating
                flower.other_uses_rating = pfaf_data.other_uses_rating
                flower.weed_potential = pfaf_data.weed_potential
                flower.care_info = pfaf_data.care_info or flower.care_info
                await db.commit()
            scraped.append("pfaf")
        except Exception:
            failed.append("pfaf")

    # Wikipedia
    if isinstance(wiki_res, BaseException) or wiki_res is None:
        failed.append("wikipedia")
    else:
        try:
            wiki_data = wiki_res
            await _upsert_source(db, flower_id, "wikipedia",
                raw_content=wiki_data.extract,
                parsed={
                    "url": wiki_data.url,
                    "summary": wiki_data.summary,
                    "taxonomy": wiki_data.taxonomy,
                })
            flower = await db.get(Flower, flower_id)
            if flower:
                flower.wikipedia_url = wiki_data.url
                await db.commit()
            scraped.append("wikipedia")
        except Exception:
            failed.append("wikipedia")

    # Wikidata
    if isinstance(wd_res, BaseException) or wd_res is None:
        failed.append("wikidata")
    else:
        try:
            wd_data = wd_res
            await _upsert_source(db, flower_id, "wikidata",
                raw_content=None,
                parsed={
                    "qid": wd_data.qid,
                    "native_range": wd_data.native_range,
                    "conservation_status": wd_data.conservation_status,
                    "family": wd_data.family,
                    "common_names": wd_data.common_names,
                    "native_range_description": wd_data.native_range_description,
                })
            scraped.append("wikidata")
        except Exception:
            failed.append("wikidata")

    # GBIF
    if isinstance(gbif_res, BaseException) or gbif_res is None:
        failed.append("gbif")
    else:
        try:
            gbif_data = gbif_res
            await _upsert_source(db, flower_id, "gbif",
                raw_content=None,
                parsed={
                    "kingdom": gbif_data.kingdom,
                    "family": gbif_data.family,
                    "genus": gbif_data.genus,
                    "species": gbif_data.species,
                    "taxonomic_status": gbif_data.taxonomic_status,
                    "habitats": gbif_data.habitats,
                    "distributions": gbif_data.distributions[:20],
                    "vernacular_names": gbif_data.vernacular_names,
                })
            scraped.append("gbif")
        except Exception:
            failed.append("gbif")

    # Update flower status
    flower = await db.get(Flower, flower_id)
    if flower:
        flower.status = "scraped" if scraped else "scrape_failed"
        await db.commit()

    return ScrapeResult(
        flower_id=flower_id,
        latin_name=latin_name,
        sources_scraped=scraped,
        sources_failed=failed,
    )


async def _upsert_source(
    db: AsyncSession,
    flower_id: int,
    source: str,
    raw_content: str | None,
    parsed: dict | None,
) -> None:
    existing = await db.execute(
        select(RawSource).where(
            RawSource.flower_id == flower_id,
            RawSource.source == source,
        )
    )
    row = existing.scalar_one_or_none()
    if row:
        row.raw_content = raw_content
        row.parsed_content = parsed
    else:
        db.add(RawSource(
            flower_id=flower_id,
            source=source,
            raw_content=raw_content,
            parsed_content=parsed,
        ))
    await db.commit()
