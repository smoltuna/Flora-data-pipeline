import asyncio
from contextlib import asynccontextmanager

import structlog
from config import settings
from database import async_session_factory, create_tables
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from log_config import configure_logging
from models import Flower
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import make_asgi_app
from routers import export, flowers
from services.observability import register_funnel_gauge, setup_observability
from sqlalchemy import func, select

log = structlog.get_logger()

# Updated every FUNNEL_REFRESH_S by a background task; read synchronously
# by the OTel observable gauge callback.
_funnel_counts: dict[str, int] = {}
FUNNEL_REFRESH_S = 15.0


async def _refresh_funnel_loop() -> None:
    while True:
        try:
            async with async_session_factory() as session:
                rows = await session.execute(
                    select(Flower.status, func.count()).group_by(Flower.status)
                )
                _funnel_counts.clear()
                _funnel_counts.update({status: int(n) for status, n in rows.all()})
        except Exception as exc:
            log.warning("funnel.refresh_failed", error=str(exc))
        await asyncio.sleep(FUNNEL_REFRESH_S)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await create_tables()
    register_funnel_gauge(lambda: dict(_funnel_counts))
    task = asyncio.create_task(_refresh_funnel_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(
    title="Flora Asset Pipeline",
    description="Automated botanical data and image pipeline for the Flora iOS app",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry — traces (OTLP → Tempo) + metrics (Prometheus pull) + MLflow sink.
setup_observability(metrics_exporter="prometheus_pull")

FastAPIInstrumentor.instrument_app(app)

# Prometheus scrape endpoint (consumed by the OTel metrics reader above)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Routers
app.include_router(flowers.router, prefix="/flowers", tags=["flowers"])
app.include_router(export.router, prefix="/export", tags=["export"])


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "environment": settings.environment}
