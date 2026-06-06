"""All flower-scoped endpoints: CRUD + data pipeline + image pipeline + image serving."""
from __future__ import annotations

from pathlib import Path

from database import async_session_factory, get_db
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from models import Flower
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tasks.pipeline import run_pipeline

router = APIRouter()


class FlowerCreate(BaseModel):
    latin_name: str
    common_name: str | None = None


class FlowerOut(BaseModel):
    id: int
    latin_name: str
    common_name: str | None
    status: str
    description: str | None
    fun_fact: str | None
    wiki_description: str | None
    habitat: str | None
    etymology: str | None
    cultural_info: str | None
    petal_color_hex: str | None
    care_info: dict | None
    edibility_rating: int | None
    medicinal_rating: int | None
    other_uses_rating: int | None
    weed_potential: str | None
    info_image_path: str | None
    info_image_author: str | None
    main_image_path: str | None
    lock_image_path: str | None
    feature_year: int | None
    feature_month: int | None
    feature_day: int | None
    confidence_scores: dict | None
    wikipedia_url: str | None

    model_config = {"from_attributes": True}


@router.get("", response_model=list[FlowerOut])
async def list_flowers(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[FlowerOut]:
    q = select(Flower).order_by(Flower.id).limit(limit).offset(offset)
    if status:
        q = q.where(Flower.status == status)
    result = await db.execute(q)
    return [FlowerOut.model_validate(f) for f in result.scalars().all()]


@router.post("", response_model=FlowerOut, status_code=201)
async def create_flower(body: FlowerCreate, db: AsyncSession = Depends(get_db)) -> FlowerOut:
    existing = await db.execute(select(Flower).where(Flower.latin_name == body.latin_name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Flower already exists")
    flower = Flower(latin_name=body.latin_name, common_name=body.common_name)
    db.add(flower)
    await db.commit()
    await db.refresh(flower)
    return FlowerOut.model_validate(flower)


@router.get("/{flower_id}", response_model=FlowerOut)
async def get_flower(flower_id: int, db: AsyncSession = Depends(get_db)) -> FlowerOut:
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    return FlowerOut.model_validate(flower)


@router.delete("/{flower_id}", status_code=204)
async def delete_flower(flower_id: int, db: AsyncSession = Depends(get_db)) -> None:
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    await db.delete(flower)
    await db.commit()


@router.post("/{flower_id}/data", response_model=FlowerOut)
async def run_data_pipeline(flower_id: int, db: AsyncSession = Depends(get_db)) -> FlowerOut:
    """Synchronously run scrape + embed + RAG + translate. Returns the enriched flower."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    updated = await run_pipeline(flower_id, db)
    return FlowerOut.model_validate(updated)


@router.post("/{flower_id}/images", response_model=FlowerOut)
async def run_image_pipeline(flower_id: int, db: AsyncSession = Depends(get_db)) -> FlowerOut:
    """Synchronously run the image pipeline (Wikimedia → FAL → rembg → lock)."""
    from config import settings
    from services.images.lock_gen import generate_lock_image
    from services.images.processor import process_info_image, process_main_image
    from services.images.wikimedia import find_images

    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")
    if flower.status not in ("enriched", "images_done", "complete"):
        raise HTTPException(
            status_code=400,
            detail=f"Flower must be enriched first (status: {flower.status})",
        )

    pair = await find_images(flower.latin_name)
    info_path, author = await process_info_image(pair.info, flower.latin_name)
    flower.info_image_path = info_path
    flower.info_image_author = author

    main_path, _ = await process_main_image(
        pair.blossom,
        flower.latin_name,
        candidates=pair.blossom_candidates,
        fal_key=settings.fal_key,
    )
    flower.main_image_path = main_path

    flower.lock_image_path = await generate_lock_image(
        main_path, flower.latin_name, fal_key=settings.fal_key
    )
    flower.status = "images_done"
    await db.commit()
    await db.refresh(flower)
    return FlowerOut.model_validate(flower)


@router.get("/{flower_id}/images/{image_type}")
async def serve_image(
    flower_id: int,
    image_type: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """Serve a processed image file (image_type: info | main | lock)."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    path_map = {
        "info": flower.info_image_path,
        "main": flower.main_image_path,
        "lock": flower.lock_image_path,
    }
    if image_type not in path_map:
        raise HTTPException(status_code=400, detail="image_type must be info, main, or lock")

    slug = path_map[image_type]
    if not slug:
        raise HTTPException(status_code=404, detail=f"No {image_type} image for this flower")

    xcassets = Path(__file__).parents[2] / "output" / "FlowerAssets.xcassets"
    candidates = [
        xcassets / f"{slug}.imageset" / "home.png",
        xcassets / f"{slug}.imageset" / "info.jpg",
    ]
    for p in candidates:
        if p.exists():
            media_type = "image/png" if p.suffix == ".png" else "image/jpeg"
            return FileResponse(p, media_type=media_type)
    raise HTTPException(status_code=404, detail="Image file not found on disk")
