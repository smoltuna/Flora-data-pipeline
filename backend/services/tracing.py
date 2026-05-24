"""Pipeline observability: per-step latency, token counts, and LLM call counts.

Usage in pipeline.py:
    tracer = PipelineTracer(flower_id, flower.latin_name)
    with tracer.step("scrape") as m:
        result = await _do_scrape(...)
        m.api_calls = 4
    tracer.log_summary()

Token counts are captured via both the side-channel counter in
services.llm._token_counter (auto-accumulated by providers) and the
LLMResponse.tokens_used return value (Session 4).
"""
from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()


@dataclass
class StepMetrics:
    name: str
    duration_s: float = 0.0
    tokens_used: int = 0      # prompt + completion tokens from complete() calls
    llm_calls: int = 0        # number of complete() calls
    api_calls: int = 0        # HTTP requests (scrapers, embed calls)
    chunks_in: int = 0
    chunks_out: int = 0
    errors: list[str] = field(default_factory=list)


class PipelineTracer:
    def __init__(self, flower_id: int, latin_name: str) -> None:
        self.flower_id = flower_id
        self.latin_name = latin_name
        self._steps: list[StepMetrics] = []
        self._start = time.perf_counter()

    @contextmanager
    def step(self, name: str) -> Generator[StepMetrics, None, None]:
        """Track a pipeline step. Captures duration and LLM token usage automatically.

        Usage:
            with tracer.step("grade") as m:
                result = await grade_retrieval(...)
                m.chunks_in = len(chunks)  # set manual counters on m
        """
        from services.llm import _token_counter
        _token_counter.read_and_reset()  # clear any stale counts from setup code
        metrics = StepMetrics(name=name)
        t0 = time.perf_counter()
        try:
            yield metrics
        except Exception as exc:
            metrics.errors.append(str(exc))
            raise
        finally:
            tokens, calls = _token_counter.read_and_reset()
            metrics.duration_s = round(time.perf_counter() - t0, 3)
            metrics.tokens_used += tokens
            metrics.llm_calls += calls
            self._steps.append(metrics)

    def summary(self) -> dict:
        """Full trace as a JSON-serializable dict."""
        total_s = round(time.perf_counter() - self._start, 3)
        return {
            "flower_id": self.flower_id,
            "latin_name": self.latin_name,
            "total_duration_s": total_s,
            "total_tokens": sum(s.tokens_used for s in self._steps),
            "total_llm_calls": sum(s.llm_calls for s in self._steps),
            "total_api_calls": sum(s.api_calls for s in self._steps),
            "steps": [
                {
                    "name": s.name,
                    "duration_s": s.duration_s,
                    "tokens_used": s.tokens_used,
                    "llm_calls": s.llm_calls,
                    "api_calls": s.api_calls,
                    "chunks_in": s.chunks_in,
                    "chunks_out": s.chunks_out,
                    "errors": s.errors,
                }
                for s in self._steps
            ],
        }

    def log_summary(self) -> None:
        """Log trace via structlog and optionally emit metrics to MLflow."""
        data = self.summary()
        log.info(
            "pipeline.trace",
            flower_id=self.flower_id,
            latin_name=self.latin_name,
            total_s=data["total_duration_s"],
            total_tokens=data["total_tokens"],
            total_llm_calls=data["total_llm_calls"],
            steps={s["name"]: s["duration_s"] for s in data["steps"]},
        )
        try:
            import mlflow
            per_step: dict[str, float] = {}
            for s in data["steps"]:
                per_step[f"trace_{s['name']}_s"] = s["duration_s"]
                if s["tokens_used"]:
                    per_step[f"trace_{s['name']}_tokens"] = float(s["tokens_used"])
                if s["llm_calls"]:
                    per_step[f"trace_{s['name']}_llm_calls"] = float(s["llm_calls"])
            mlflow.log_metrics({
                "trace_total_tokens": float(data["total_tokens"]),
                "trace_total_llm_calls": float(data["total_llm_calls"]),
                "trace_total_s": data["total_duration_s"],
                **per_step,
            })
        except Exception:
            pass
