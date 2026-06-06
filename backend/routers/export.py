"""Export endpoint — trigger xcassets export for a flower or all complete flowers."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from database import get_db
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from models import Flower, RawSource, Translation
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Default xcassets output directory (project_root/output/FlowerAssets.xcassets)
_DEFAULT_XCASSETS_DIR = Path(__file__).parents[2] / "output" / "FlowerAssets.xcassets"

router = APIRouter()


# ---------------------------------------------------------------------------
# Care info — canonical icon+label mapping
# ---------------------------------------------------------------------------

_CARE_LABEL_MAP: dict[str, dict] = {
    # ── Sun / shade ──────────────────────────────────────────────────────────
    "full sun": {"icon": "sun.max.fill", "label": "Full Sun"},
    "sun": {"icon": "sun.max.fill", "label": "Full Sun"},
    "no shade": {"icon": "sun.max.fill", "label": "Full Sun"},
    "full shade": {"icon": "moon.fill", "label": "Full Shade"},
    "deep shade": {"icon": "moon.fill", "label": "Full Shade"},
    "dense shade": {"icon": "moon.fill", "label": "Full Shade"},
    "part shade": {"icon": "cloud.sun.fill", "label": "Part Shade"},
    "partial shade": {"icon": "cloud.sun.fill", "label": "Part Shade"},
    "semi-shade": {"icon": "cloud.sun.fill", "label": "Part Shade"},
    "semi shade": {"icon": "cloud.sun.fill", "label": "Part Shade"},
    "dappled shade": {"icon": "cloud.sun.fill", "label": "Part Shade"},
    "light shade": {"icon": "cloud.sun.fill", "label": "Part Shade"},
    # ── Soil / moisture ──────────────────────────────────────────────────────
    "well drained": {"icon": "drop", "label": "Well Drained"},
    "well-drained": {"icon": "drop", "label": "Well Drained"},
    "well drained soil": {"icon": "drop", "label": "Well Drained Soil"},
    "well-drained soil": {"icon": "drop", "label": "Well Drained Soil"},
    "moist": {"icon": "drop.fill", "label": "Moist Soil"},
    "moist soil": {"icon": "drop.fill", "label": "Moist Soil"},
    "moisture retentive": {"icon": "drop.fill", "label": "Moist Soil"},
    "wet": {"icon": "drop.fill", "label": "Wet Soil"},
    "wet soil": {"icon": "drop.fill", "label": "Wet Soil"},
    "boggy": {"icon": "drop.fill", "label": "Wet Soil"},
    "waterlogged": {"icon": "drop.fill", "label": "Wet Soil"},
    "water plants": {"icon": "drop.fill", "label": "Water Plants"},
    "aquatic": {"icon": "drop.fill", "label": "Water Plants"},
    "pond": {"icon": "drop.fill", "label": "Water Plants"},
    # ── Hardiness ────────────────────────────────────────────────────────────
    "fully hardy": {"icon": "snowflake", "label": "Fully Hardy"},
    "frost hardy": {"icon": "snowflake", "label": "Frost Hardy"},
    "frost resistant": {"icon": "snowflake", "label": "Frost Hardy"},
    "half hardy": {"icon": "snowflake", "label": "Half Hardy"},
    "tender": {"icon": "snowflake", "label": "Tender"},
    "not hardy": {"icon": "snowflake", "label": "Tender"},
    "tropical": {"icon": "snowflake", "label": "Tender"},
    "subtropical": {"icon": "snowflake", "label": "Tender"},
}

_SKIP_KEYS = frozenset([
    "cultivation details", "cultivation", "propagation", "edibility",
    "medicinal", "other uses", "edible", "weed potential", "habitats",
    "notes", "synonyms", "family",
])

_SKIP_VALUES = frozenset([
    "not specified", "n/a", "unknown", "information not available",
    "usda", "none", "no information",
])


def _match_care_value(text: str) -> dict | None:
    """Match a single text string to a canonical {icon, label}, or None."""
    v = text.strip().lower()
    if not v or v in _SKIP_VALUES:
        return None
    if v in _CARE_LABEL_MAP:
        return _CARE_LABEL_MAP[v]
    best: tuple[int, dict] | None = None
    for key, entry in _CARE_LABEL_MAP.items():
        if key in v or v in key:
            if best is None or len(key) > best[0]:
                best = (len(key), entry)
    return best[1] if best else None


def _normalize_care_info(care_info) -> list[dict]:
    """Normalise care_info to the canonical [{icon, label}] list."""
    if not care_info:
        return []

    if isinstance(care_info, list):
        valid = [
            item for item in care_info
            if isinstance(item, dict) and "icon" in item and "label" in item
        ]
        if valid:
            return valid

    if isinstance(care_info, dict):
        result: list[dict] = []
        seen_labels: set[str] = set()

        def _add(entry: dict | None) -> None:
            if entry and entry["label"] not in seen_labels:
                result.append(entry)
                seen_labels.add(entry["label"])

        for raw_key, raw_val in care_info.items():
            key = raw_key.strip().lower()
            if any(skip in key for skip in _SKIP_KEYS):
                continue
            parts = [p.strip() for p in str(raw_val).split(",") if p.strip()]
            for part in parts:
                entry = _match_care_value(part)
                if entry:
                    _add(entry)
                    continue
                if part.lower() in ("yes", "true", "y", "1"):
                    _add(_match_care_value(key))
                    continue
                _add(_match_care_value(f"{key} {part}"))

        return result

    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_stem(latin_name: str) -> str:
    """'Iris germanica' → 'iris-germanica'"""
    return latin_name.replace("×", "x").replace(" ", "-").lower()


async def _fetch_pfaf_care_info(flower_id: int, db: AsyncSession) -> dict | None:
    """Return the raw care_info dict from the PFAF raw_source, or None."""
    result = await db.execute(
        select(RawSource).where(
            RawSource.flower_id == flower_id,
            RawSource.source == "pfaf",
        )
    )
    src = result.scalar_one_or_none()
    if src and src.parsed_content:
        ci = src.parsed_content.get("care_info")
        if ci:
            return ci
    return None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class ExportResult(BaseModel):
    exported: int
    output_path: str


@router.get("/{flower_id}")
async def export_flower(flower_id: int, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """Return the Flora-compatible JSON payload for a single flower from the xcassets bundle."""
    flower = await db.get(Flower, flower_id)
    if not flower:
        raise HTTPException(status_code=404, detail="Flower not found")

    if flower.status not in ("enriched", "images_done", "complete"):
        raise HTTPException(status_code=400, detail="Flower not yet enriched")

    flowers_json = _DEFAULT_XCASSETS_DIR / "flowers.dataset" / "flowers.json"
    if not flowers_json.exists():
        raise HTTPException(
            status_code=503,
            detail="xcassets bundle not built yet — run POST /export/xcassets first",
        )

    payloads = json.loads(flowers_json.read_text())
    entry = next((p for p in payloads if p.get("latinName") == flower.latin_name), None)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"{flower.latin_name!r} not in bundle — run POST /export/xcassets first",
        )

    return JSONResponse(content=entry)


@router.post("/batch", response_model=ExportResult)
async def export_batch(output_dir: str = "/tmp/flora_export") -> ExportResult:
    """Write individual flower JSON files from the xcassets bundle."""
    flowers_json = _DEFAULT_XCASSETS_DIR / "flowers.dataset" / "flowers.json"
    if not flowers_json.exists():
        raise HTTPException(
            status_code=503,
            detail="xcassets bundle not built yet — run POST /export/xcassets first",
        )

    payloads = json.loads(flowers_json.read_text())
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for payload in payloads:
        filename = payload["latinName"].replace(" ", "_").lower() + ".json"
        (out_path / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    return ExportResult(exported=len(payloads), output_path=str(out_path))


@router.post("/xcassets", response_model=ExportResult)
async def export_xcassets(
    output_dir: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> ExportResult:
    """Write the complete FlowerAssets.xcassets bundle (Contents.json + flowers.dataset/)."""
    xcassets_dir = Path(output_dir) if output_dir else _DEFAULT_XCASSETS_DIR
    count = await build_xcassets_bundle(db, xcassets_dir)
    return ExportResult(exported=count, output_path=str(xcassets_dir))


async def build_xcassets_bundle(db: AsyncSession, xcassets_dir: Path) -> int:
    """Build the complete FlowerAssets.xcassets bundle from all enriched flowers.

    Creates:
      {xcassets_dir}/Contents.json
      {xcassets_dir}/flowers.dataset/Contents.json
      {xcassets_dir}/flowers.dataset/flowers.json  ← array of all flower payloads

    Returns the number of flowers exported.
    """
    result = await db.execute(
        select(Flower).where(Flower.status.in_(["enriched", "images_done", "complete"]))
    )
    flowers = result.scalars().all()

    payloads: list[dict] = []
    for flower in flowers:
        trans_result = await db.execute(
            select(Translation).where(Translation.flower_id == flower.id)
        )
        translations = trans_result.scalars().all()
        pfaf_care = await _fetch_pfaf_care_info(flower.id, db)
        payloads.append(_build_payload(flower, translations, pfaf_care))

    _write_xcassets_files(xcassets_dir, payloads)
    return len(payloads)


def _write_xcassets_files(xcassets_dir: Path, payloads: list[dict]) -> None:
    """Write xcassets bundle structure to disk (synchronous file I/O)."""
    xcassets_dir.mkdir(parents=True, exist_ok=True)

    # Top-level Contents.json (required by Xcode to recognise the bundle)
    (xcassets_dir / "Contents.json").write_text(
        json.dumps({"info": {"author": "xcode", "version": 1}}, indent=2)
    )

    dataset_dir = xcassets_dir / "flowers.dataset"
    dataset_dir.mkdir(exist_ok=True)

    (dataset_dir / "Contents.json").write_text(json.dumps({
        "data": [{"filename": "flowers.json", "idiom": "universal"}],
        "info": {"author": "xcode", "version": 1},
    }, indent=2))

    (dataset_dir / "flowers.json").write_text(
        json.dumps(payloads, indent=2, ensure_ascii=False)
    )


def _build_payload(
    flower: Flower,
    translations: list[Translation] | Sequence[Translation],
    pfaf_care: dict | None = None,
) -> dict:
    """Build Flora-compatible JSON matching flowers.json schema."""
    trans_map: dict[str, Translation] = {t.language: t for t in translations}
    stem = _image_stem(flower.latin_name)
    care_source = pfaf_care if pfaf_care else flower.care_info

    def localized(field: str, lang: str) -> str | None:
        t = trans_map.get(lang)
        return getattr(t, field, None) if t else None

    return {
        "name": flower.common_name or flower.latin_name,
        "latinName": flower.latin_name,
        "description": flower.description or "",
        "funFact": flower.fun_fact or "",
        "petalColorHex": flower.petal_color_hex or "#FFFFFF",
        "imageName": flower.main_image_path or stem,
        "lockImageName": flower.lock_image_path or f"{stem}-lock",
        "infoImageName": flower.info_image_path or f"{stem}-info",
        "infoImageAuthor": flower.info_image_author or "",
        "careInfo": _normalize_care_info(care_source),
        "year": flower.feature_year or 0,
        "month": flower.feature_month or 0,
        "day": flower.feature_day or 0,
        "wikiDescription": flower.wiki_description or "",
        "habitat": flower.habitat or "",
        "etymology": flower.etymology or "",
        "culturalInfo": flower.cultural_info or "",
        "wikipediaUrl": flower.wikipedia_url or "",
        "translations": {
            lang: {
                k: v for k, v in {
                    "name": localized("name", lang),
                    "description": localized("description", lang),
                    "funFact": localized("fun_fact", lang),
                    "wikiDescription": localized("wiki_description", lang),
                    "habitat": localized("habitat", lang),
                    "etymology": localized("etymology", lang),
                    "culturalInfo": localized("cultural_info", lang),
                }.items() if v is not None
            }
            for lang in ("de", "fr", "es", "it", "zh", "ja")
            if lang in trans_map
        },
    }
