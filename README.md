<p align="center">
  <img src="assets/flora-logo.png" alt="Flora" width="120">
</p>

# Flora Asset Pipeline

> Automated botanical data enrichment and image pipeline for the [Flora iOS app](https://apps.apple.com/ca/app/flora-flower-of-the-day/id6759986494)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-orange.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Does

Given a plant's Latin name (e.g. `Iris germanica`), the pipeline scrapes four botanical sources, enriches and fact-checks the data through a multi-stage RAG pipeline (CRAG + Self-RAG), translates into 6 languages, and generates 3 processed images — outputting a complete `FlowerAssets.xcassets` bundle ready to drop into the iOS project.

**Output per flower:** description, fun fact, wiki summary, habitat, etymology, cultural info, petal color hex, care data, per-field confidence scores, 6-language translations, and 3 images.

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │            FastAPI Backend               │
                          │                                         │
  Latin Name  ──────────► │  /scrape   →  PFAF · Wikipedia          │
                          │             Wikidata · GBIF              │
                          │                 │                       │
                          │                 ▼                       │
                          │  Embedder  →  pgvector (HNSW 768d)      │
                          │                 │                       │
                          │  Retriever → all chunks for flower      │
                          │                 │                       │
                          │  Deduplicator (cosine ≥ 0.92)          │
                          │                 │                       │
                          │  Adaptive routing (full/sparse/minimal) │
                          │                 │                       │
                          │  CRAG Grader → per-field quality check  │
                          │                 │                       │
                          │  Synthesizer → LLM (Ollama/Groq/…)     │
                          │                 │                       │
                          │  Self-RAG Verifier → confidence scores  │
                          │                 │                       │
                          │  /translate →  Ollama llama3.2:3b       │
                          │  /images   →  Wikimedia → rembg → lock │
                          │  /export   →  FlowerAssets.xcassets     │
                          └─────────────────────────────────────────┘
                                          │
                          ┌──────────────────────────────┐
                          │      Next.js Dashboard        │
                          │  Library · Detail · Scores    │
                          └──────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A [fal.ai](https://fal.ai/dashboard/keys) API key (for vision judging + lock icon generation — pipeline degrades gracefully without it)

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/Flora-Asset-Pipeline.git
cd Flora-Asset-Pipeline
cp .env.example .env
```

Edit `.env` — at minimum set `FAL_KEY`. `LLM_PROVIDER=ollama` is the default and requires no API key.

### 2. Create the Python virtualenv

```bash
uv sync
```

This is required to run the pipeline scripts locally (they connect directly to the DB).

### 3. Start all services

```bash
# Recommended: handles pull, preflight checks, and startup order
./scripts/start_services.sh

# Or equivalently (Docker handles ordering via healthchecks):
docker compose up -d
```

Services started:
- **PostgreSQL + pgvector** → `localhost:5432`
- **Ollama** → `localhost:11434` (pulls `llama3.2:3b` + `nomic-embed-text` on first boot, ~2.5 GB)
- **MLflow** → `localhost:5001`
- **FastAPI backend** → `localhost:8000` (Swagger: `/docs`)
- **Next.js frontend** → `localhost:3000`

> **First run:** Ollama will pull both models on startup. The backend healthcheck waits up to 10 minutes — this is normal. Monitor with `docker compose logs -f ollama`.

### 4. Run the pipeline

```bash
# Full pipeline: data + images + xcassets export (recommended)
.venv/bin/python scripts/run_all.py

# Data only (no fal.ai key needed)
.venv/bin/python scripts/run_all.py --skip-images

# Images only (flowers must already be enriched)
.venv/bin/python scripts/run_all.py --skip-data

# Single flower
.venv/bin/python scripts/run_all.py --name "Rosa canina"

# First N pending flowers from the database
.venv/bin/python scripts/run_all.py --limit 5
```

Edit the `FLOWERS` list at the top of `scripts/run_all.py` to change which flowers are processed. Default demo flowers are in `data/demo_flowers.txt`.

### 5. Check the output

```
output/FlowerAssets.xcassets/
  Contents.json                        ← Xcode bundle root
  flowers.dataset/
    Contents.json
    flowers.json                       ← array of all flower records
  iris-germanica.imageset/
    Contents.json
    home.png                           ← 600×600 RGBA, bg-removed blossom
  iris-germanica-info.imageset/
    Contents.json
    info.jpg                           ← max 1024px, JPEG q85
  iris-germanica-lock.imageset/
    Contents.json
    lock.png                           ← 200×200 RGBA, FLUX-generated icon
```

This folder can be dropped directly into the Flora iOS Xcode project.

### 6. Browse the dashboard

Open [http://localhost:3000](http://localhost:3000) — inspect confidence scores, view enriched data, trigger pipeline stages manually.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_all.py` | Full pipeline: seed → data → images → xcassets export |
| `scripts/start_services.sh` | Start Docker services with preflight checks (`--no-ui`, `--stop`) |

`run_all.py` flags: `--name "Latin name"`, `--file flowers.txt`, `--limit N`, `--skip-images` (data only), `--skip-data` (images only, flowers must be enriched).

---

## Pipeline Stages

| Stage | What it does | How |
|-------|-------------|-----|
| **1. Scrape** | Fetch raw botanical data from four sources | PFAF via BeautifulSoup (HTML parse, 2 s rate-limit delay); Wikipedia via REST API + MediaWiki API for taxobox; Wikidata via SPARQL (QID lookup → conservation status, common names); GBIF v1 REST API (species match → vernacular names, distributions) |
| **2. Embed** | Convert source text to 768-dim vectors | Each source → single chunk; `nomic-embed-text` via Ollama; stored in `source_embeddings` with HNSW index (m=16, ef=64) |
| **3. Retrieve** | Fetch all embeddings for the flower | Per-flower SELECT from `source_embeddings`; ranking and filtering handled downstream |
| **4. Deduplicate** | Remove semantically near-identical chunks | Pairwise cosine similarity matrix; cosine ≥ 0.92 → same cluster; keeps longest text, tiebreaks by structured source (PFAF/Wikidata > Wikipedia prose) |
| **5. Adaptive routing** | Decide which fields to synthesize | PFAF + Wikipedia present → full 7-field synthesis; Wikidata or GBIF only → skip `fun_fact`, `cultural_info`; no recognised sources → also skip `etymology` |
| **6. CRAG Grade** | Per-field retrieval quality check | LLM asked whether chunks are relevant per field; ≥2 relevant → sufficient; 1 → partial; 0 → field set to `"Information not available."` |
| **7. Synthesize** | Generate structured JSON from sources | Source-attributed prompt grouped by label (PFAF / Wikipedia / Wikidata / GBIF); LLM returns JSON with all output fields; temperature 0.1 |
| **8. Verify (Self-RAG)** | Fact-check each generated field | LLM rates 0.0–1.0 how well the generated claim is supported by source text; score stored in `flowers.confidence_scores` JSONB |
| **9. Persist** | Save enriched flower to DB | Updates `flowers` record; status → `"enriched"`; assigns sequential feature dates starting 2026-05-01 (flower[0] = May 1, flower[1] = May 2, …) |
| **10. Translate** | Translate into 6 languages | All 6 languages (DE/FR/ES/IT/ZH/JA) via Ollama `llama3.2:3b`; field-by-field calls for reliability (42 LLM calls per flower); ~10–30 min for 5 flowers on M-series Mac |
| **Images** | Fetch and process 3 image variants | Wikimedia Commons search (CC0/CC-BY only, ≥500 px); fal.ai Llava-Next scores candidates; `rembg` for background removal; FLUX via fal.ai for lock icon; Pillow for resize/crop |
| **Export** | Build xcassets bundle | Writes `flowers.json` array + per-flower imageset folders into `output/FlowerAssets.xcassets/` |

---

## Project Structure

```
Flora-Asset-Pipeline/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Pydantic settings (env-driven)
│   ├── models.py                # SQLAlchemy ORM: Flower, RawSource, SourceEmbedding, Translation
│   ├── database.py              # Async session factory + create_tables()
│   ├── routers/                 # flowers · scrape · enrich · images · translate · export
│   ├── services/
│   │   ├── scraper/             # pfaf · wikipedia · wikidata · gbif
│   │   ├── rag/                 # embedder · retriever · deduplicator · grader · synthesizer · verifier
│   │   ├── llm/                 # provider abstraction · groq · ollama · together · gemini · rate_limiter
│   │   ├── images/              # wikimedia · processor · lock_gen
│   │   └── translation/         # translator (Ollama, all 6 languages)
│   └── tasks/
│       └── pipeline.py          # 10-stage sequential orchestrator + MLflow tracking
├── frontend/
│   └── src/
│       ├── app/                 # Dashboard + flower detail (Next.js 15 App Router)
│       ├── components/          # FlowerCard · DataFieldsView · ConfidenceScores
│       ├── lib/api.ts           # Typed API client
│       └── types/flower.ts      # TypeScript interfaces
├── tests/
│   ├── conftest.py
│   ├── test_rag_pipeline.py     # Unit: synthesis parsing, deduplication
│   ├── test_scrapers.py         # Unit: Wikipedia + GBIF with respx mocks
│   ├── test_deduplication.py    # Unit: extended deduplication scenarios
│   ├── test_confidence_and_images.py
│   └── test_integration.py      # Integration: FastAPI + PostgreSQL end-to-end
├── data/
│   └── demo_flowers.txt         # 15 botanically diverse demo entries
├── scripts/
│   ├── run_all.py               # Full pipeline (--skip-images, --skip-data, --limit)
│   └── start_services.sh        # Docker service startup with preflight checks
├── output/                      # Generated — FlowerAssets.xcassets/ written here
├── .env.example
└── docker-compose.yml           # PostgreSQL · Ollama · MLflow · backend · frontend
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11+, FastAPI, SQLAlchemy 2 async |
| **Vector DB** | PostgreSQL 17 + pgvector (HNSW index, 768d) |
| **Embedding model** | `nomic-embed-text` via Ollama (768-dim) |
| **LLM (synthesis)** | Provider-agnostic — Ollama (default), Groq, Together.ai, OpenAI, Gemini |
| **LLM (translation)** | Ollama `llama3.2:3b` — all 6 languages, field-by-field |
| **RAG patterns** | CRAG, Self-RAG, semantic deduplication, adaptive routing |
| **Image processing** | rembg (background removal), Pillow (resize/crop), fal.ai (vision judge + FLUX lock icon) |
| **Frontend** | Next.js 15 App Router, TypeScript, Tailwind CSS |
| **Observability** | MLflow experiment tracking, structlog |
| **Infrastructure** | Docker Compose (local) |

---

## LLM Providers

Configure via `LLM_PROVIDER` in `.env`. All providers implement the same `LLMProvider` protocol. Ollama is the default and requires no API key.

| Provider | `LLM_PROVIDER` | Env var | Notes |
|----------|---------------|---------|-------|
| **Ollama** (default) | `ollama` | — | Runs locally; no cost; slower on CPU/integrated GPU |
| Groq | `groq` | `GROQ_API_KEY` | Fast; free tier has ~28 RPM — built-in rate limiter handles this |
| Together.ai | `together` | `TOGETHER_API_KEY` | Good fallback if Groq quota is exhausted |
| OpenAI | `openai` | `OPENAI_API_KEY` | Highest quality; paid |
| Gemini | `gemini` | `GEMINI_API_KEY` | Free tier has very low RPM limits |

Embeddings always use Ollama (`nomic-embed-text`) regardless of the synthesis provider.

> **Translation note:** `TRANSLATION_PROVIDER` is a separate setting (default: `ollama`). Avoid switching it to Groq on the free tier — the 42 sequential calls per flower quickly exhaust the TPM limit.

---

## API Reference

Swagger docs auto-generated at [http://localhost:8000/docs](http://localhost:8000/docs).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/flowers` | GET | List flowers (filter by `?status=`) |
| `/flowers` | POST | Add a new flower |
| `/flowers/{id}` | GET | Flower detail |
| `/scrape/{id}/sync` | POST | Scrape all four sources (sync) |
| `/enrich/{id}/sync` | POST | Run full RAG pipeline (sync) |
| `/enrich/{id}/chunks` | GET | Inspect retrieved + deduplicated chunks |
| `/images/{id}` | POST | Run image pipeline |
| `/translate/{id}` | POST | Translate into all 6 languages |
| `/export/{id}` | GET | Export xcassets-compatible JSON for one flower |
| `/export/batch` | POST | Export all enriched flowers as individual JSON files |
| `/export/xcassets` | POST | Build complete `FlowerAssets.xcassets` bundle |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus scrape endpoint |

---

## Running Tests

```bash
# Unit tests (no DB required)
.venv/bin/python -m pytest tests/ -v -m "not integration"

# Integration tests (requires running PostgreSQL)
DATABASE_URL=postgresql+asyncpg://flora:flora@localhost:5432/flora \
  .venv/bin/python -m pytest tests/test_integration.py -v -m integration
```

---

## MLflow Experiment Tracking

Each pipeline run logs to the `flora-enrichment` experiment, making it easy to compare LLM providers, spot low-confidence flowers, and track how source coverage affects output quality across batches.

- **Tags:** `latin_name`, `llm_provider`, `flower_id`
- **Metrics:** `pipeline_duration_s`, `chunks_retrieved`, `chunks_after_dedup`, per-field `confidence_llm_*`

View the MLflow UI at [http://localhost:5001](http://localhost:5001).

---

## Design Decisions

- **No LangChain** — all RAG components are plain Python for full control and simpler debugging
- **Sequential pipeline** — no parallelism within a flower; easier to trace, reason about, and test
- **Provider-agnostic LLM** — swap Ollama → Groq → Together with a single env var change
- **Ollama for translation** — avoids rate limits and cost; field-by-field calls trade speed for reliability with small models
- **Semantic dedup threshold 0.92** — calibrated to collapse paraphrases without losing complementary information from different sources
- **Self-RAG confidence scores** — stored per field, enabling downstream filtering of low-confidence outputs
- **CC0/CC-BY images only** — no licensing friction for Flora iOS app distribution

---

## License

MIT
