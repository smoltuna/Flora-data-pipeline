"""RAG-grounded translation service.

Strategy:
  - `name` field: try Wikidata P1843 / GBIF vernacular names from the DB first
    (already scraped, no extra HTTP). If unavailable, ask the LLM with the
    English source as context. If the result fails sanity checks, fall back to
    the Latin name — we never persist made-up names.
  - Body fields: each is translated with a separate plain-text call grounded
    in the English source text. Sanity checks drop rejected fields rather than
    persisting garbage.
  - Provider: defaults to ollama (step="translation"), so TRANSLATION_MODEL=qwen2.5:7b
    in .env routes translation through a stronger multilingual model without
    touching the rest of the pipeline.
"""
from __future__ import annotations

import asyncio
import re

import structlog
from models import Flower, RawSource, Translation
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.llm.provider import get_provider


def _get_translation_provider():
    from config import settings
    return get_provider(settings.translation_provider, step="translation")


log = structlog.get_logger()

ALL_LANGUAGES = {"de", "fr", "es", "it", "zh", "ja"}

LANG_NAMES = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
}

TEXT_FIELDS = [
    "description", "fun_fact", "wiki_description",
    "habitat", "etymology", "cultural_info",
]

_FIELD_PROMPT_KEY = {
    "description": "description",
    "fun_fact": "funFact",
    "wiki_description": "wikiDescription",
    "habitat": "habitat",
    "etymology": "etymology",
    "cultural_info": "culturalInfo",
}
_REVERSE_KEY = {v: k for k, v in _FIELD_PROMPT_KEY.items()}

MAX_FIELD_CHARS = 600
MAX_NAME_CHARS = 60
MIN_BODY_CHARS = 20
MIN_NAME_CHARS = 2

# Scripts: ja/zh outputs must contain CJK; de/fr/es/it must NOT be mostly CJK.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[぀-ヿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# Phrases the small LLM tends to leak into output (translator notes, planning).
# Two layers:
#   1) "Nota:"/"Note:" + colon at line start — English/Romance translator notes.
#   2) Chinese instruction/error phrases qwen sometimes emits mid-output, like
#      "原句中断" ("sentence interrupted"), "原消息已结束" ("message ended"),
#      "请提供" ("please provide"). These appear inside DE/FR/IT fields when qwen
#      thinks in Chinese mid-generation. Catch any of these substrings.
_META_LEAK_RE = re.compile(
    r"(?:^|\n)\s*(?:Nota|Note|Notes|Traduzione|Translator|Translator's note)\s*[:：\-]"
    r"|(?:^|\W)(?:translation note|Latin translation|備考|译注|翻译注|译者注|訳注)(?:[:：\s]|$)"
    r"|原句|原消息|原文|请提供|请检查|无法完成|未能提供|保持不变|特别注意|翻译中",
    re.IGNORECASE,
)

# Bullet/list markers — body text should be prose, not a checklist
_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)


async def translate_flower(flower_id: int, session: AsyncSession) -> None:
    """Translate all text fields + common name for a flower into all supported languages."""
    flower = await session.get(Flower, flower_id)
    if not flower:
        return

    scraped_names = await _load_scraped_common_names(session, flower_id)

    async def _safe_translate(lang: str) -> tuple[str, dict[str, str | None] | None]:
        log.info("translation.start", flower_id=flower_id, lang=lang)
        try:
            fields = await _get_fields(flower, lang, scraped_names.get(lang))
            return lang, fields
        except Exception as e:
            log.error("translation.error", flower_id=flower_id, lang=lang, error=str(e))
            return lang, None

    results = await asyncio.gather(*[_safe_translate(lang) for lang in sorted(ALL_LANGUAGES)])

    for lang, fields in results:
        if fields is not None:
            await _upsert_translation(
                session, flower_id, lang, fields, source_method="llm_translation"
            )


# GBIF returns ISO 639-3 (3-letter) language codes; the rest of the pipeline
# uses ISO 639-1 (2-letter). Map the languages we care about.
_ISO3_TO_ISO1 = {
    "deu": "de", "ger": "de",
    "fra": "fr", "fre": "fr",
    "spa": "es",
    "ita": "it",
    "jpn": "ja",
    "zho": "zh", "chi": "zh", "cmn": "zh",
    "eng": "en",
}


async def _load_scraped_common_names(
    session: AsyncSession, flower_id: int
) -> dict[str, str]:
    """Read Wikidata P1843 + GBIF vernacular names for this flower from the DB.

    Returns {lang: name} merged across sources, preferring Wikidata (canonical
    labels) over GBIF (community-curated). Language codes are normalised to
    ISO 639-1 (de/fr/es/it/ja/zh) — GBIF emits 639-3 so we translate them.
    """
    result = await session.execute(
        select(RawSource).where(
            RawSource.flower_id == flower_id,
            RawSource.source.in_(("wikidata", "gbif")),
        )
    )
    rows = result.scalars().all()

    gbif_names: dict[str, str] = {}
    wikidata_names: dict[str, str] = {}
    for row in rows:
        parsed = row.parsed_content or {}
        if row.source == "wikidata":
            names = parsed.get("common_names") or {}
            if isinstance(names, dict):
                wikidata_names = {
                    k: v for k, v in names.items()
                    if isinstance(v, str) and v.strip()
                }
        elif row.source == "gbif":
            names = parsed.get("vernacular_names") or {}
            if isinstance(names, dict):
                for raw_lang, name in names.items():
                    if not (isinstance(name, str) and name.strip()):
                        continue
                    lang_key = _ISO3_TO_ISO1.get(raw_lang, raw_lang)
                    # Don't overwrite an existing GBIF entry — first match wins
                    gbif_names.setdefault(lang_key, name)

    merged = {**gbif_names, **wikidata_names}  # wikidata wins on key collisions
    return merged


async def _get_fields(
    flower: Flower, lang: str, scraped_name: str | None
) -> dict[str, str | None]:
    """Run LLM translation for one language, with scraped-name lookup + sanity checks."""
    llm = _get_translation_provider()
    lang_name = LANG_NAMES[lang]

    source: dict[str, str] = {}
    for field in TEXT_FIELDS:
        text = getattr(flower, field, None)
        if text and text != "Information not available.":
            source[_FIELD_PROMPT_KEY[field]] = text[:MAX_FIELD_CHARS]

    common = flower.common_name or flower.latin_name
    body_only = {k: v for k, v in source.items() if k != "name"}

    translated = await _fieldwise_translate(
        llm, flower.latin_name, common, lang, lang_name, body_only,
    )

    # ── Resolve `name` field: Wikidata/GBIF → sanity-checked LLM → Latin ──
    name = _resolve_name(
        scraped_name=scraped_name,
        llm_name=translated.get("name"),
        lang=lang,
        latin_name=flower.latin_name,
    )

    # ── Sanity-check body fields, drop rejected ──
    fields: dict[str, str | None] = {"name": name}
    for prompt_key, field_name in _REVERSE_KEY.items():
        val = translated.get(prompt_key)
        if val:
            val = val.strip()
        if val and _sanity_check_body(val, lang):
            fields[field_name] = val
        else:
            if val:
                log.info(
                    "translation.field_rejected",
                    flower_id=flower.id, lang=lang, field=field_name,
                    preview=val[:80],
                )
            fields[field_name] = None

    n = len([v for v in fields.values() if v])
    log.info("translation.done", flower_id=flower.id, lang=lang, n_fields=n)
    return fields


def _resolve_name(
    scraped_name: str | None, llm_name: str | None, lang: str, latin_name: str
) -> str:
    """Pick a name in priority order. Never returns hallucinated junk.

    Order: Wikidata/GBIF scraped name → sanity-passing LLM name → Latin name.
    """
    if scraped_name and _sanity_check_name(scraped_name, lang):
        return scraped_name.strip()
    if llm_name:
        llm_name = llm_name.strip().strip("\"'.,;:")
        if _sanity_check_name(llm_name, lang):
            return llm_name
        log.info("translation.name_rejected", lang=lang, preview=llm_name[:60])
    return latin_name


async def _fieldwise_translate(
    llm,
    latin_name: str,
    common_name: str,
    lang: str,
    lang_name: str,
    source: dict[str, str],
) -> dict:
    """Translate name + each body field, grounding every prompt in the English source."""
    results: dict[str, str] = {}

    # Name: ask for the LOCAL common name only, not a calque of the English name.
    try:
        resp = await llm.complete(
            prompt=(
                f"Provide the local common name in {lang_name} for the plant "
                f"{latin_name} (English: {common_name}).\n\n"
                f"Rules:\n"
                f"- Reply with ONLY the {lang_name} name, no quotes, no parens, "
                f"no explanation.\n"
                f"- Use the established {lang_name} botanical common name, not a "
                f"word-for-word translation.\n"
                f"- If no established {lang_name} common name exists, reply exactly: "
                f"{latin_name}\n"
            ),
            system=f"You are a botanical expert. Reply with only the plant name in {lang_name}.",
        )
        name = resp.text.strip().strip("\"'")
        if name:
            results["name"] = name
    except Exception as e:
        log.warning("translation.name_failed", lang=lang_name, error=str(e))

    # Body fields: each gets the English source as the ONLY source.
    for prompt_key, text in source.items():
        try:
            resp = await llm.complete(
                prompt=(
                    f"Translate the following botanical text about {latin_name} "
                    f"into {lang_name}.\n\n"
                    f"English source:\n{text}\n\n"
                    f"Rules:\n"
                    f"- Translate faithfully — do not add, remove, or invent facts.\n"
                    f"- Keep proper nouns ({latin_name}) in Latin.\n"
                    f"- Reply with ONLY the {lang_name} translation as prose. "
                    f"No bullets, no notes, no commentary.\n"
                ),
                system=(
                    f"You are a botanical translator. "
                    f"Reply with only the translation in {lang_name}."
                ),
            )
            val = resp.text.strip()
            if val:
                results[prompt_key] = val
        except Exception as e:
            log.warning("translation.field_failed", field=prompt_key, lang=lang_name, error=str(e))

    return results


def _sanity_check_name(value: str, lang: str) -> bool:
    """Reject names that are too short, too long, wrong-script, or contain meta-commentary."""
    if not value:
        return False
    v = value.strip()
    if len(v) < MIN_NAME_CHARS or len(v) > MAX_NAME_CHARS:
        return False
    # No leaked meta-commentary
    if _META_LEAK_RE.search(v):
        return False
    # Names are short — multi-line responses are wrong
    if "\n" in v:
        return False
    # Script check: ja/zh names must be either pure CJK or pure Latin (the
    # latin-name fallback). Mixed scripts (e.g. 'ばらすいkusen', 'ツルイチゴ…andesia')
    # are LLM garbage and must be rejected.
    if lang in ("ja", "zh"):
        cjk = bool(_CJK_RE.search(v))
        latin = bool(_LATIN_RE.search(v))
        if cjk and latin:
            return False
        if not cjk and not latin:
            return False
    else:
        # Latin-script langs: should be mostly Latin letters, not mostly CJK
        cjk = len(_CJK_RE.findall(v))
        latin = len(_LATIN_RE.findall(v))
        if cjk > latin:
            return False
    # Reject if value is only punctuation/whitespace
    if not re.search(r"\w", v):
        return False
    return True


def _sanity_check_body(value: str, lang: str) -> bool:
    """Reject body text that's too short, wrong-script, bulleted, or contains meta-commentary."""
    if not value:
        return False
    v = value.strip()
    if len(v) < MIN_BODY_CHARS:
        return False
    if _META_LEAK_RE.search(v):
        return False
    if _BULLET_RE.search(v):
        return False
    if lang in ("ja", "zh"):
        # Must contain CJK
        if not _CJK_RE.search(v):
            return False
        # Japanese must have hiragana or katakana — pure-kanji is usually Chinese mis-tag
        if lang == "ja" and not _HIRAGANA_KATAKANA_RE.search(v):
            return False
    else:
        # Latin-script langs: ZERO tolerance for CJK. Any CJK character means
        # qwen is leaking Chinese mid-output (commentary, error msgs, or
        # mis-translated fragments). Drop the field rather than persist mixed text.
        if _CJK_RE.search(v):
            return False
    return True


async def _upsert_translation(
    session: AsyncSession,
    flower_id: int,
    lang: str,
    fields: dict[str, str | None],
    source_method: str,
) -> None:
    existing = await session.execute(
        select(Translation).where(
            Translation.flower_id == flower_id,
            Translation.language == lang,
        )
    )
    row = existing.scalar_one_or_none()
    if row:
        for k, v in fields.items():
            setattr(row, k, v)
        row.source_method = source_method
    else:
        session.add(Translation(
            flower_id=flower_id,
            language=lang,
            source_method=source_method,
            **fields,
        ))
    await session.commit()
