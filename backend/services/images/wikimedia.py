"""Wikimedia Commons image search for plant photographs.

Search strategy — cast a wide net, then score aggressively:
  1. Category:{latin_name}          (most precise; well-curated for many species)
  2. Category:{genus}               (broader genus-level category)
  3. Text search: "{latin_name}"    (catches mis-categorised files)
  4. Text search: "{genus} flower"  (last-resort fallback)

All queries run until we accumulate >= MIN_CANDIDATES images, then stop.
Each candidate is a WikimediaImage dataclass ready for unified scoring
in the search orchestrator (search.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

_API = "https://commons.wikimedia.org/w/api.php"
_HEADERS = {
    "User-Agent": "FloraRAGPipeline/1.0 (portfolio; contact: simone.84858@gmail.com)"
}

_ALLOWED_LICENSES = frozenset({
    "cc0", "cc-by", "cc by", "cc-by-sa", "cc by-sa",
    "public domain", "pd",
})

_ACCEPTED_MIME = frozenset({"image/jpeg", "image/png"})

# Min candidates before we stop issuing new search queries
_MIN_CANDIDATES = 40

# ---------------------------------------------------------------------------
# Skip patterns — non-photographic or wrong-subject content
# ---------------------------------------------------------------------------

_SKIP_RE = re.compile(
    r"illustration|drawing|painting|watercolor|lithograph|engraving|sketch"
    r"|herbarium|specimen|pressed|dried"
    r"|stamp|postage|colnect|rcin|barcode"
    r"|museum|naturalis"
    r"|distribution|range|\bmap\b"
    r"|\blogo\b|\bicon\b|\bclipart\b|\bdiagram\b"
    # Commercial / non-botanical contexts
    r"|\bposter\b|\bstore\b|\bshop\b|\bcollage\b|\bpanel\b|\bpackaging\b|\blabel\b"
    # Camera dump filenames
    r"|\bbatch\b"
    # Buildings, monuments
    r"|abbaye|abbey|cathedral|eglise|church|castle|château|chateau"
    r"|monastery|basilica|mosque|temple\b|chapel"
    r"|\bbouquet\b|\barrangement\b|\bvase\b|\bpot\b"
    r"|panorami|panorama|landscape\b"
    # Animals / insects on flowers
    r"|bombus|apis|butterfly|moth|bee\b|bumblebee|insect|beetle|spider|bird"
    r"|caterpillar|larvae|larva|hymenoptera|lepidoptera|coleoptera|diptera"
    r"|pollinator|pollinat"
    # Common butterfly genera
    r"|\bvanessa\b|\bpapilio\b|\bpieris\b|\bgonepteryx\b|\baglais\b"
    # Person-centred photos
    r"|\bgirl\b|\bwoman\b|\bman\b|\bboy\b|\bchild\b|\bperson\b|\bpeople\b"
    r"|\bdívka\b|\bchica\b|\bfrau\b|\bhomme\b|\bfemme\b"
    r"|\d{7,}",  # long numeric IDs
    re.IGNORECASE,
)

_DESC_SKIP_RE = re.compile(
    r"herbarium sheet|pressed plant|dried specimen|botanical illustration"
    r"|line drawing|pen and ink|woodcut"
    r"|\bposter\b|\bstore\b|\bshop\b|\bcollage\b"
    r"|collection of (?:flowers|plants)|mixed flowers",
    re.IGNORECASE,
)


@dataclass
class WikimediaImage:
    title: str
    url: str
    author: str
    license: str
    width: int
    height: int
    size_bytes: int
    description: str = ""
    categories: str = ""
    thumb_url: str = ""  # CDN thumbnail URL (1024 px) from iiurlwidth — preferred for downloads

    @property
    def aspect(self) -> float:
        """width / height."""
        return self.width / self.height if self.height else 1.0

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def _text(self) -> str:
        """Combined searchable text for scoring."""
        return f"{self.title} {self.description} {self.categories}"

    @property
    def source(self) -> str:
        return "wikimedia"


@dataclass
class ImagePair:
    info: WikimediaImage              # artistic / landscape — for the detail/info screen
    blossom: WikimediaImage           # top-scored close-up (Gemini will re-rank)
    blossom_candidates: list | None = None   # top N candidates for Gemini vision to judge


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_image(page: dict) -> WikimediaImage | None:
    """Validate and convert an API page dict to a WikimediaImage."""
    info_list = page.get("imageinfo", [])
    if not info_list:
        return None
    info = info_list[0]
    meta = info.get("extmetadata", {})

    # License
    raw_license = (
        meta.get("LicenseShortName", {}).get("value", "") or ""
    ).lower()
    if not any(lic in raw_license for lic in _ALLOWED_LICENSES):
        return None

    # MIME — raster only
    mime = info.get("mime", "").lower()
    if mime not in _ACCEPTED_MIME:
        return None

    # Dimensions — at least 500 px short side
    width = info.get("width", 0)
    height = info.get("height", 0)
    if min(width, height) < 500:
        return None

    # File size: skip stubs and huge scans
    size = info.get("size", 0)
    if not (30_000 <= size <= 15_000_000):
        return None

    title = page.get("title", "")
    if _SKIP_RE.search(title):
        return None

    description = re.sub(
        r"<[^>]+>", "",
        meta.get("ImageDescription", {}).get("value", "") or "",
    ).strip()
    categories = (
        meta.get("Categories", {}).get("value", "") or ""
    ).replace("|", " ")

    if _DESC_SKIP_RE.search(description):
        return None

    author_raw = (
        meta.get("Artist", {}).get("value")
        or meta.get("Credit", {}).get("value")
        or "Unknown"
    )
    author = re.sub(r"<[^>]+>", "", author_raw).strip() or "Unknown"

    return WikimediaImage(
        title=title,
        url=info.get("url", ""),
        thumb_url=info.get("thumburl", ""),
        author=author,
        license=raw_license,
        width=width,
        height=height,
        size_bytes=size,
        description=description,
        categories=categories,
    )


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

_IMAGEINFO_PARAMS = {
    "prop": "imageinfo",
    "iiprop": "url|size|extmetadata|mime",
    "iiurlwidth": 1024,   # request blessed CDN thumbnail URL to avoid 429s
    "format": "json",
}


async def _category_search(
    client: httpx.AsyncClient, category: str, limit: int = 50,
) -> list[WikimediaImage]:
    """Fetch images from a Wikimedia Commons category."""
    params: dict[str, object] = {
        **_IMAGEINFO_PARAMS,
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmnamespace": 6,
        "gcmlimit": limit,
    }
    resp = await client.get(_API, params=params)  # type: ignore[arg-type]
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return [img for p in pages.values() if (img := _parse_image(p))]


async def _text_search(
    client: httpx.AsyncClient, query: str, limit: int = 40,
) -> list[WikimediaImage]:
    """Full-text search for images on Wikimedia Commons."""
    params: dict[str, object] = {
        **_IMAGEINFO_PARAMS,
        "action": "query",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": query,
        "gsrlimit": limit,
    }
    resp = await client.get(_API, params=params)  # type: ignore[arg-type]
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return [img for p in pages.values() if (img := _parse_image(p))]


# ---------------------------------------------------------------------------
# Public API — search Wikimedia only (called by search.py orchestrator)
# ---------------------------------------------------------------------------

async def search_wikimedia(latin_name: str) -> list[WikimediaImage]:
    """Search Wikimedia Commons and return all valid candidates.

    Does NOT score or select — that's handled by the unified orchestrator.
    """
    genus = latin_name.split()[0]

    async with httpx.AsyncClient(timeout=30.0, headers=_HEADERS) as client:
        candidates: list[WikimediaImage] = []
        seen: set[str] = set()

        def _add(imgs: list[WikimediaImage]) -> None:
            for img in imgs:
                if img.title not in seen and img.url:
                    seen.add(img.title)
                    candidates.append(img)

        # 1. Species category — most precise
        _add(await _category_search(client, latin_name, limit=50))

        # 2. Genus category — broader but catches subspecies / cultivars
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _category_search(client, genus, limit=40))

        # 3. Full-text search with exact Latin name
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _text_search(client, f'"{latin_name}"', limit=40))

        # 4. Genus + "flower" text search
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _text_search(client, f"{genus} flower", limit=30))

        # 5. Genus + "blossom close-up" — specifically targeting blossom shots
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _text_search(client, f"{genus} blossom close-up", limit=20))

        # 6. Plain genus text search as last resort
        if len(candidates) < _MIN_CANDIDATES:
            _add(await _text_search(client, genus, limit=30))

    return candidates

