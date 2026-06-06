"""Orchestrates all scrapers for a single flower — runs them concurrently and persists results."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from models import Flower, RawSource
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.scraper.gbif import fetch_gbif
from services.scraper.pfaf import scrape_pfaf
from services.scraper.web_search import search_web
from services.scraper.wikidata import fetch_wikidata
from services.scraper.wikipedia import fetch_wikipedia


@dataclass
class ScrapeResult:
    flower_id: int
    latin_name: str
    sources_scraped: list[str]
    sources_failed: list[str]


async def scrape_all_sources(
    flower_id: int, latin_name: str, db: AsyncSession
) -> ScrapeResult:
    """Run all 5 scrapers concurrently, persist results, return summary."""
    scraped: list[str] = []
    failed: list[str] = []

    pfaf_res, wiki_res, wd_res, gbif_res, web_res = await asyncio.gather(
        scrape_pfaf(latin_name),
        fetch_wikipedia(latin_name),
        fetch_wikidata(latin_name),
        fetch_gbif(latin_name),
        search_web(latin_name),
        return_exceptions=True,
    )

    # PFAF
    if isinstance(pfaf_res, BaseException) or pfaf_res is None:
        failed.append("pfaf")
    else:
        try:
            await _upsert_source(db, flower_id, "pfaf",
                raw_content=pfaf_res.raw_text,
                parsed={
                    "common_name": pfaf_res.common_name,
                    "edibility_rating": pfaf_res.edibility_rating,
                    "medicinal_rating": pfaf_res.medicinal_rating,
                    "other_uses_rating": pfaf_res.other_uses_rating,
                    "weed_potential": pfaf_res.weed_potential,
                    "habitat": pfaf_res.habitat,
                    "care_info": pfaf_res.care_info,
                })
            flower = await db.get(Flower, flower_id)
            if flower:
                if pfaf_res.common_name and not flower.common_name:
                    flower.common_name = pfaf_res.common_name
                flower.edibility_rating = pfaf_res.edibility_rating
                flower.medicinal_rating = pfaf_res.medicinal_rating
                flower.other_uses_rating = pfaf_res.other_uses_rating
                flower.weed_potential = pfaf_res.weed_potential
                flower.care_info = pfaf_res.care_info or flower.care_info
                await db.commit()
            scraped.append("pfaf")
        except Exception:
            failed.append("pfaf")

    # Wikipedia
    if isinstance(wiki_res, BaseException) or wiki_res is None:
        failed.append("wikipedia")
    else:
        try:
            await _upsert_source(db, flower_id, "wikipedia",
                raw_content=wiki_res.extract,
                parsed={
                    "url": wiki_res.url,
                    "summary": wiki_res.summary,
                    "taxonomy": wiki_res.taxonomy,
                })
            flower = await db.get(Flower, flower_id)
            if flower:
                flower.wikipedia_url = wiki_res.url
                await db.commit()
            scraped.append("wikipedia")
        except Exception:
            failed.append("wikipedia")

    # Wikidata
    if isinstance(wd_res, BaseException) or wd_res is None:
        failed.append("wikidata")
    else:
        try:
            await _upsert_source(db, flower_id, "wikidata",
                raw_content=None,
                parsed={
                    "qid": wd_res.qid,
                    "native_range": wd_res.native_range,
                    "conservation_status": wd_res.conservation_status,
                    "family": wd_res.family,
                    "common_names": wd_res.common_names,
                    "native_range_description": wd_res.native_range_description,
                })
            scraped.append("wikidata")
        except Exception:
            failed.append("wikidata")

    # GBIF
    if isinstance(gbif_res, BaseException) or gbif_res is None:
        failed.append("gbif")
    else:
        try:
            await _upsert_source(db, flower_id, "gbif",
                raw_content=None,
                parsed={
                    "kingdom": gbif_res.kingdom,
                    "family": gbif_res.family,
                    "genus": gbif_res.genus,
                    "species": gbif_res.species,
                    "taxonomic_status": gbif_res.taxonomic_status,
                    "habitats": gbif_res.habitats,
                    "distributions": gbif_res.distributions[:20],
                    "vernacular_names": gbif_res.vernacular_names,
                })
            scraped.append("gbif")
        except Exception:
            failed.append("gbif")

    # Web search (multiple results stored under indexed source names)
    if isinstance(web_res, BaseException):
        failed.append("web_ddg")
    else:
        web_results = web_res or []
        stored_count = 0
        for i, result in enumerate(web_results):
            try:
                await _upsert_source(
                    db, flower_id, f"web_ddg_{i}",
                    raw_content=result.content,
                    parsed={"url": result.url, "title": result.title, "score": result.score},
                )
                stored_count += 1
            except Exception:
                pass
        if stored_count > 0:
            scraped.append("web_ddg")

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
            flower_id=flower_id, source=source,
            raw_content=raw_content, parsed_content=parsed,
        ))
    await db.commit()
