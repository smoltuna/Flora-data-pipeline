"""OpenTelemetry tracing setup + custom sinks.

One collector (OTel SDK) fanning out to three sinks:
  - Jaeger UI            via OTLPSpanExporter (gRPC, OTEL_EXPORTER_OTLP_ENDPOINT)
  - MLflow run metrics   via _MLflowSpanProcessor (forwards span attrs as metrics)
  - In-memory summary    via _BatchSummaryProcessor (only registered for CLI runs)

Public API:
  setup_observability(include_batch_summary=False) — idempotent init
  get_tracer()                                     — module-scoped tracer
  step_span(name)                                  — pipeline-step ctx manager
  batch_summary                                    — read after a CLI batch
  shutdown_observability()                         — flush pending OTLP exports
"""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from opentelemetry import metrics, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import CallbackOptions, Counter, Histogram, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.trace import Span

MetricsExporter = Literal["prometheus_pull", "otlp_push", "none"]

_initialized = False
_tracer: trace.Tracer | None = None
_meter_provider: MeterProvider | None = None

# Lazy-init Prometheus instruments — created on first use so the global meter
# provider (set up by setup_observability) is guaranteed ready.
_meter_initialized = False
_pipeline_duration: Histogram | None = None
_pipelines_completed: Counter | None = None
_pipelines_failed: Counter | None = None
_step_duration: Histogram | None = None
_llm_tokens_total: Counter | None = None
_llm_calls_total: Counter | None = None
_field_confidence: Histogram | None = None
_crag_grades: Counter | None = None


def _ensure_meters() -> None:
    global _meter_initialized, _pipeline_duration, _pipelines_completed, _pipelines_failed
    global _step_duration, _llm_tokens_total, _llm_calls_total
    global _field_confidence, _crag_grades
    if _meter_initialized:
        return
    meter = metrics.get_meter("flora-pipeline")
    _pipeline_duration = meter.create_histogram(
        "flora_pipeline_duration",
        unit="s",
        description="End-to-end pipeline duration per flower",
    )
    _pipelines_completed = meter.create_counter(
        "flora_pipelines_completed",
        description="Pipelines that reached 'enriched' status",
    )
    _pipelines_failed = meter.create_counter(
        "flora_pipelines_failed",
        description="Pipelines that ended in 'failed' status",
    )
    _step_duration = meter.create_histogram(
        "flora_pipeline_step_duration",
        unit="s",
        description="Per-step pipeline duration (scrape/embed/grade/...)",
    )
    _llm_tokens_total = meter.create_counter(
        "flora_llm_tokens",
        description="LLM tokens consumed, labelled by step",
    )
    _llm_calls_total = meter.create_counter(
        "flora_llm_calls",
        description="LLM calls issued, labelled by step",
    )
    _field_confidence = meter.create_histogram(
        "flora_field_confidence",
        description="Self-RAG verifier confidence per field [0,1]",
    )
    _crag_grades = meter.create_counter(
        "flora_crag_grades",
        description="CRAG retrieval grade outcomes per field",
    )
    _meter_initialized = True


def record_pipeline_completion(duration_s: float) -> None:
    """Record one completed pipeline run."""
    _ensure_meters()
    assert _pipeline_duration is not None and _pipelines_completed is not None
    _pipeline_duration.record(duration_s)
    _pipelines_completed.add(1)


def record_pipeline_failure() -> None:
    """Record one pipeline run that ended in 'failed' status."""
    _ensure_meters()
    assert _pipelines_failed is not None
    _pipelines_failed.add(1)


def record_field_confidence(field: str, score: float) -> None:
    _ensure_meters()
    assert _field_confidence is not None
    _field_confidence.record(score, {"field": field})


def record_crag_grade(field: str, grade: str) -> None:
    """grade ∈ {sufficient, partial, insufficient}."""
    _ensure_meters()
    assert _crag_grades is not None
    _crag_grades.add(1, {"field": field, "grade": grade})


def register_funnel_gauge(get_counts: Callable[[], dict[str, int]]) -> None:
    """Register an observable gauge that reports flower counts by status.

    The callback is invoked synchronously by the OTel SDK on each collection,
    so get_counts() must be cheap and non-blocking. The caller is responsible
    for keeping the underlying dict fresh (e.g. via an async background task
    in the API lifespan).
    """
    _ensure_meters()
    meter = metrics.get_meter("flora-pipeline")

    def _callback(_: CallbackOptions) -> Iterator[Observation]:
        for status, count in get_counts().items():
            yield Observation(count, {"status": status})

    meter.create_observable_gauge(
        "flora_flowers_by_status",
        callbacks=[_callback],
        description="Current count of flowers per pipeline status",
    )


def _record_llm_usage(step: str, tokens: int, calls: int) -> None:
    _ensure_meters()
    assert _llm_tokens_total is not None and _llm_calls_total is not None
    if tokens:
        _llm_tokens_total.add(tokens, {"step": step})
    if calls:
        _llm_calls_total.add(calls, {"step": step})


class _MLflowSpanProcessor(SpanProcessor):
    """Forward each span's duration + LLM metrics to the active MLflow run, if any."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return

    def on_end(self, span: ReadableSpan) -> None:
        try:
            import mlflow
            if not mlflow.active_run():
                return
            duration_s = (span.end_time - span.start_time) / 1_000_000_000.0
            name = span.name
            attrs = span.attributes or {}

            if name == "flower":
                trace_id = span.context.trace_id
                f = batch_summary.flowers.get(trace_id, {})
                mlflow.log_metrics({
                    "trace_total_s": duration_s,
                    "trace_total_tokens": float(f.get("total_tokens", 0)),
                    "trace_total_llm_calls": float(f.get("total_llm_calls", 0)),
                })
                return

            metrics: dict[str, float] = {f"trace_{name}_s": duration_s}
            for key in ("tokens_used", "llm_calls", "api_calls", "chunks_in", "chunks_out"):
                v = attrs.get(key)
                if v:
                    # Normalize "tokens_used" → "tokens" for backward-compatible metric names.
                    metric_key = "tokens" if key == "tokens_used" else key
                    metrics[f"trace_{name}_{metric_key}"] = float(v)
            mlflow.log_metrics(metrics)
        except Exception:
            pass

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _BatchSummaryProcessor(SpanProcessor):
    """In-memory per-flower step collector backing the CLI ASCII summary table.

    Children always end before their parent span, so when the "flower" span fires
    on_end we already hold every step's aggregates.

    Note: this grows unbounded — only register it for short-lived CLI runs, not
    the long-running FastAPI service.
    """

    def __init__(self) -> None:
        self.flowers: dict[int, dict[str, Any]] = defaultdict(
            lambda: {"steps": {}, "total_s": 0.0, "total_llm_calls": 0, "total_tokens": 0}
        )

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return

    def on_end(self, span: ReadableSpan) -> None:
        trace_id = span.context.trace_id
        duration_s = (span.end_time - span.start_time) / 1_000_000_000.0
        attrs = span.attributes or {}
        entry = self.flowers[trace_id]

        if span.name == "flower":
            entry["latin_name"] = attrs.get("latin_name", "?")
            entry["flower_id"] = attrs.get("flower_id")
            entry["total_s"] = duration_s
            return

        tokens = int(attrs.get("tokens_used", 0) or 0)
        calls = int(attrs.get("llm_calls", 0) or 0)
        entry["steps"][span.name] = {
            "duration_s": duration_s,
            "tokens": tokens,
            "llm_calls": calls,
        }
        entry["total_llm_calls"] += calls
        entry["total_tokens"] += tokens

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def snapshot(self) -> list[dict]:
        """Per-flower dicts, in the order their flower span ended."""
        return [
            {
                "latin_name": f.get("latin_name", "?"),
                "flower_id": f.get("flower_id"),
                "total_duration_s": f["total_s"],
                "total_llm_calls": f["total_llm_calls"],
                "total_tokens": f["total_tokens"],
                "steps": [
                    {"name": name, **data}
                    for name, data in f["steps"].items()
                ],
            }
            for f in self.flowers.values()
            if "latin_name" in f
        ]


batch_summary = _BatchSummaryProcessor()


def _build_meter_provider(
    resource: Resource, mode: MetricsExporter
) -> MeterProvider | None:
    """Build a MeterProvider whose reader matches the deployment mode.

    prometheus_pull → PrometheusMetricReader (long-running API; scraped via /metrics).
    otlp_push       → PeriodicExportingMetricReader → Prometheus OTLP receiver
                      (short-lived CLI; flushed on shutdown).
    none            → no MeterProvider; metric calls become no-ops.
    """
    if mode == "none":
        return None

    if mode == "prometheus_pull":
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        return MeterProvider(resource=resource, metric_readers=[PrometheusMetricReader()])

    if mode == "otlp_push":
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        endpoint = os.getenv(
            "PROMETHEUS_OTLP_ENDPOINT",
            "http://localhost:9090/api/v1/otlp/v1/metrics",
        )
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint),
            export_interval_millis=5_000,
        )
        return MeterProvider(resource=resource, metric_readers=[reader])

    raise ValueError(f"Unknown metrics_exporter mode: {mode}")


def setup_observability(
    *,
    include_batch_summary: bool = False,
    metrics_exporter: MetricsExporter = "none",
) -> None:
    """Initialize OTel tracer + metrics + sinks. Safe to call multiple times.

    include_batch_summary=True registers the in-memory aggregator used by the
    CLI summary table. Leave False for long-lived servers (FastAPI).

    metrics_exporter controls how metrics leave the process. The API uses
    "prometheus_pull"; the CLI uses "otlp_push" so the same counters/histograms
    reach the same Prometheus regardless of entry point.
    """
    global _initialized, _tracer, _meter_provider
    if _initialized:
        return
    resource = Resource.create({"service.name": "flora-pipeline"})

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(insecure=True)))
    if include_batch_summary:
        provider.add_span_processor(batch_summary)
    provider.add_span_processor(_MLflowSpanProcessor())
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("flora-pipeline")

    _meter_provider = _build_meter_provider(resource, metrics_exporter)
    if _meter_provider is not None:
        metrics.set_meter_provider(_meter_provider)

    _initialized = True


def get_tracer() -> trace.Tracer:
    if not _initialized:
        setup_observability()
    assert _tracer is not None
    return _tracer


def shutdown_observability() -> None:
    """Flush pending OTLP exports (traces and metrics)."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()
    if _meter_provider is not None:
        _meter_provider.shutdown()


@contextmanager
def step_span(name: str) -> Iterator[Span]:
    """OTel span for one pipeline step, with auto LLM token capture.

    Resets the module-global _token_counter on entry, reads it on exit, and
    attaches the totals as span attributes. The counter is module-global, so
    concurrent flowers (asyncio.gather) can bleed token counts across each
    other — pre-existing limitation, not introduced by OTel.

    Also emits the per-step duration histogram and step-labelled token/call
    counters so dashboards can break work down by stage.
    """
    import time as _time

    from services.llm import _token_counter

    _token_counter.read_and_reset()
    start = _time.perf_counter()
    with get_tracer().start_as_current_span(name) as span:
        try:
            yield span
        finally:
            tokens, calls = _token_counter.read_and_reset()
            if tokens:
                span.set_attribute("tokens_used", tokens)
            if calls:
                span.set_attribute("llm_calls", calls)
            _record_llm_usage(name, tokens, calls)
            _ensure_meters()
            assert _step_duration is not None
            _step_duration.record(_time.perf_counter() - start, {"step": name})
