from contextlib import asynccontextmanager

from config import settings
from database import create_tables
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from log_config import configure_logging
from opentelemetry import metrics as otel_metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import make_asgi_app
from routers import export, flowers
from services.observability import setup_observability


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await create_tables()
    yield


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

# OpenTelemetry — traces (OTLP → Jaeger) + MLflow sink. See services/observability.py.
setup_observability()

# OpenTelemetry — metrics exported to Prometheus
_prometheus_reader = PrometheusMetricReader()
_meter_provider = MeterProvider(metric_readers=[_prometheus_reader])
otel_metrics.set_meter_provider(_meter_provider)

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
