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

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.trace import Span

_initialized = False
_tracer: trace.Tracer | None = None


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


def setup_observability(*, include_batch_summary: bool = False) -> None:
    """Initialize OTel tracer + sinks. Safe to call multiple times.

    include_batch_summary=True registers the in-memory aggregator used by the
    CLI summary table. Leave False for long-lived servers (FastAPI).
    """
    global _initialized, _tracer
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
    _initialized = True


def get_tracer() -> trace.Tracer:
    if not _initialized:
        setup_observability()
    assert _tracer is not None
    return _tracer


def shutdown_observability() -> None:
    """Flush pending OTLP exports."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()


@contextmanager
def step_span(name: str) -> Iterator[Span]:
    """OTel span for one pipeline step, with auto LLM token capture.

    Resets the module-global _token_counter on entry, reads it on exit, and
    attaches the totals as span attributes. The counter is module-global, so
    concurrent flowers (asyncio.gather) can bleed token counts across each
    other — pre-existing limitation, not introduced by OTel.
    """
    from services.llm import _token_counter

    _token_counter.read_and_reset()
    with get_tracer().start_as_current_span(name) as span:
        try:
            yield span
        finally:
            tokens, calls = _token_counter.read_and_reset()
            if tokens:
                span.set_attribute("tokens_used", tokens)
            if calls:
                span.set_attribute("llm_calls", calls)
