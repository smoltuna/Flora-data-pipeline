import asyncio

import httpx
from config import settings

from services.llm.rate_limiter import gemini_limiter

_GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
_DEFAULT_MODEL = "gemini-2.5-flash"
_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4


class GeminiProvider:
    def __init__(self) -> None:
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")

    async def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        backoff = 5.0
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            await gemini_limiter.acquire()

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        _GEMINI_CHAT_URL,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": _DEFAULT_MODEL,
                            "messages": messages,
                            "temperature": 0.1,
                        },
                    )

                if response.status_code in _RETRYABLE_CODES and attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                from services.llm import _token_counter
                _token_counter.record(
                    usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    # OpenAI-compatible responses may contain segmented text content.
                    content = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )

                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("Gemini API returned empty completion content")

                return content

            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= _MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

        raise RuntimeError("Gemini API: max retries exceeded")

    async def embed(self, text: str) -> list[float]:
        # Gemini embeddings are not needed in this pipeline.
        from services.llm.ollama import OllamaProvider

        return await OllamaProvider().embed(text)
