import asyncio

import httpx
from config import settings

from services.llm.provider import LLMResponse

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_DEFAULT_MODEL = "llama-3.1-8b-instant"
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4


class GroqProvider:
    def __init__(self) -> None:
        self.api_key = settings.groq_api_key
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set")

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        from services.llm.rate_limiter import groq_limiter

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        backoff = 2.0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            await groq_limiter.acquire()

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        _GROQ_CHAT_URL,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": _DEFAULT_MODEL,
                            "messages": messages,
                            "temperature": 0.1,
                        },
                    )

                if response.status_code in _RETRYABLE_CODES and attempt < _MAX_ATTEMPTS:
                    retry_after = response.headers.get("retry-after")
                    wait = min(float(retry_after), 30.0) if retry_after else backoff
                    await asyncio.sleep(wait)
                    backoff = min(backoff * 2, 30.0)
                    continue

                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                from services.llm import _token_counter
                _token_counter.record(tokens)
                content = data["choices"][0]["message"]["content"]

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Groq API returned empty completion content")

                return LLMResponse(text=content, tokens_used=tokens)

            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= _MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        raise RuntimeError("Groq API: max retries exceeded")
