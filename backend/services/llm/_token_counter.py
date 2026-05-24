"""Side-channel token counter for LLM providers.

Each provider calls record() inside complete() after a successful HTTP response.
PipelineTracer resets the counter at the start of each step and reads it at the end.

Providers also return tokens via LLMResponse.tokens_used (Session 4), but this
side-channel remains in use by PipelineTracer for automatic per-step aggregation.
"""
_tokens: int = 0
_calls: int = 0


def record(tokens: int) -> None:
    """Accumulate tokens and increment call count."""
    global _tokens, _calls
    _tokens += tokens
    _calls += 1


def read_and_reset() -> tuple[int, int]:
    """Return (tokens, calls) accumulated since last reset, then zero both."""
    global _tokens, _calls
    t, c = _tokens, _calls
    _tokens = 0
    _calls = 0
    return t, c
