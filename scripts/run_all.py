"""End-to-end pipeline: seed → data (scrape/RAG/translate) → images → xcassets.

Edit the FLOWERS list below or pass overrides on the command line.

Usage:
  uv run python scripts/run_all.py                  # process FLOWERS list
  uv run python scripts/run_all.py --name "Rosa canina"
  uv run python scripts/run_all.py --file flowers.txt
  uv run python scripts/run_all.py --limit 5        # first N pending flowers in DB
  uv run python scripts/run_all.py --skip-images    # data pipeline only
  uv run python scripts/run_all.py --skip-data      # images only (flowers must be enriched)
"""
from __future__ import annotations

# ── Flowers to process ───────────────────────────────────────────────────────
FLOWERS = [
    "Iris germanica",
    "Papaver orientale"
    # "Nymphaea alba"
]
# ─────────────────────────────────────────────────────────────────────────────

# Feature dates: flower[0] gets START_DATE, flower[1] gets START_DATE+1, etc.
from datetime import date, timedelta
START_DATE = date(2026, 5, 1)

import asyncio
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import structlog  # noqa: E402
from sqlalchemy import select  # noqa: E402

from config import settings  # noqa: E402
from database import async_session_factory, create_tables  # noqa: E402
from log_config import configure_logging  # noqa: E402
from models import Flower  # noqa: E402
from tasks.pipeline import run_pipeline  # noqa: E402
from services.images.wikimedia import find_images  # noqa: E402
from services.images.processor import process_info_image, process_main_image  # noqa: E402
from services.images.lock_gen import generate_lock_image  # noqa: E402
from routers.export import build_xcassets_bundle  # noqa: E402

log = structlog.get_logger()


async def _seed(latin_names: list[str]) -> None:
    """Insert flowers that are not yet in the database (idempotent)."""
    async with async_session_factory() as session:
        added = 0
        for name in latin_names:
            name = name.strip()
            if not name:
                continue
            existing = await session.execute(select(Flower).where(Flower.latin_name == name))
            if existing.scalar_one_or_none() is None:
                session.add(Flower(latin_name=name, status="pending"))
                added += 1
        await session.commit()
    if added:
        log.info("seed.done", added=added)


async def _run_images(flower: Flower, session) -> None:
    pair = await find_images(flower.latin_name)

    info_slug, author = await process_info_image(pair.info, flower.latin_name)
    flower.info_image_path = info_slug       # e.g. "iris-germanica-info"
    flower.info_image_author = author

    main_file_path, _ = await process_main_image(
        pair.blossom,
        flower.latin_name,
        candidates=pair.blossom_candidates,
        fal_key=settings.fal_key,
    )
    # DB stores just the slug, not the filesystem path
    flower.main_image_path = flower.latin_name.replace(" ", "-").lower()

    lock_slug = await generate_lock_image(
        main_file_path, flower.latin_name, fal_key=settings.fal_key
    )
    flower.lock_image_path = lock_slug       # e.g. "iris-germanica-lock"

    flower.status = "images_done"
    await session.commit()


ENRICHED_STATUSES = ("enriched", "images_done", "complete")


async def _load_flowers(
    latin_names: list[str] | None,
    limit: int | None = None,
    skip_data: bool = False,
) -> list[Flower]:
    async with async_session_factory() as session:
        if latin_names:
            result = await session.execute(
                select(Flower).where(Flower.latin_name.in_(latin_names))
            )
        elif skip_data:
            q = (
                select(Flower)
                .where(Flower.status.in_(ENRICHED_STATUSES))
                .order_by(Flower.id)
            )
            if limit:
                q = q.limit(limit)
            result = await session.execute(q)
        else:
            q = select(Flower).where(Flower.status == "pending").order_by(Flower.id)
            if limit:
                q = q.limit(limit)
            result = await session.execute(q)
        return result.scalars().all()


def _eta(elapsed: float, done: int, total: int) -> float | None:
    if done <= 0:
        return None
    remaining = total - done
    return round((elapsed / done) * remaining, 1) if remaining > 0 else 0.0


async def main(
    latin_names: list[str] | None,
    skip_images: bool,
    skip_data: bool = False,
    limit: int | None = None,
) -> None:
    configure_logging()
    await create_tables()

    if latin_names and not skip_data:
        await _seed(latin_names)

    flowers = await _load_flowers(latin_names, limit=limit, skip_data=skip_data)
    total = len(flowers)

    if total == 0:
        log.info("run_all.nothing_to_do")
        return

    log.info("run_all.start", n_flowers=total, skip_images=skip_images, skip_data=skip_data)
    data_ok = data_fail = img_ok = img_fail = 0
    batch_start = time.perf_counter()

    for index, flower in enumerate(flowers, start=1):
        elapsed = time.perf_counter() - batch_start
        feature_date = START_DATE + timedelta(days=index - 1)
        log.info(
            "run_all.processing",
            latin_name=flower.latin_name,
            progress=f"{index}/{total}",
            feature_date=str(feature_date),
            eta_s=_eta(elapsed, data_ok + data_fail + img_ok + img_fail, total),
        )

        # ── Stage 1: Data pipeline ──────────────────────────────────────────
        if not skip_data:
            step_start = time.perf_counter()
            async with async_session_factory() as session:
                try:
                    await run_pipeline(flower.id, session, feature_date=feature_date)
                    data_ok += 1
                    log.info(
                        "run_all.data_done",
                        latin_name=flower.latin_name,
                        elapsed_s=round(time.perf_counter() - step_start, 1),
                    )
                except Exception as exc:
                    data_fail += 1
                    log.error(
                        "run_all.data_error",
                        latin_name=flower.latin_name,
                        error=str(exc),
                        exc_type=type(exc).__name__,
                    )
                    continue  # skip images if data failed

        if skip_images:
            continue

        # ── Stage 2: Image pipeline ─────────────────────────────────────────
        step_start = time.perf_counter()
        async with async_session_factory() as session:
            try:
                f = await session.get(Flower, flower.id)
                if f.status not in ENRICHED_STATUSES:
                    log.warning(
                        "run_all.images_skip",
                        latin_name=flower.latin_name,
                        reason=f"status={f.status!r} — must be enriched first",
                    )
                    continue
                await _run_images(f, session)
                img_ok += 1
                log.info(
                    "run_all.images_done",
                    latin_name=flower.latin_name,
                    elapsed_s=round(time.perf_counter() - step_start, 1),
                )
            except Exception as exc:
                img_fail += 1
                log.error(
                    "run_all.images_error",
                    latin_name=flower.latin_name,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )

    # ── Stage 3: xcassets bundle export ────────────────────────────────────────
    xcassets_dir = Path(__file__).parent.parent / "output" / "FlowerAssets.xcassets"
    log.info("run_all.exporting_bundle", output=str(xcassets_dir))
    try:
        async with async_session_factory() as session:
            n_exported = await build_xcassets_bundle(session, xcassets_dir)
        log.info("run_all.export_done", n_flowers=n_exported, output=str(xcassets_dir))
        print(f"\nExported {n_exported} flower(s) to:\n  {xcassets_dir}")
    except Exception as exc:
        log.error("run_all.export_error", error=str(exc))
        print(f"\nBundle export failed: {exc}")

    total_elapsed = round(time.perf_counter() - batch_start, 1)
    log.info(
        "run_all.complete",
        data_succeeded=data_ok,
        data_failed=data_fail,
        images_succeeded=img_ok,
        images_failed=img_fail,
        total=total,
        elapsed_s=total_elapsed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end Flora pipeline")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--name", type=str, help="Single Latin name")
    source.add_argument("--file", type=Path, help="Text file with one Latin name per line")
    parser.add_argument(
        "--limit",
        type=int,
        help="Process the first N matching flowers in the database",
    )
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--skip-images",
        action="store_true",
        help="Run data pipeline only (scrape/RAG/translate), skip image generation",
    )
    stage.add_argument(
        "--skip-data",
        action="store_true",
        help="Run image pipeline only (flowers must already be enriched)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.name:
        names: list[str] | None = [args.name]
    elif args.file:
        names = args.file.read_text().splitlines()
    elif args.limit or args.skip_data:
        names = None
    else:
        names = FLOWERS

    asyncio.run(main(names, skip_images=args.skip_images, skip_data=args.skip_data, limit=args.limit))
