import httpx
from config import settings

from services.llm.provider import LLMResponse

_TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
_DEFAULT_LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"


class TogetherProvider:
    def __init__(self) -> None:
        self.api_key = settings.together_api_key
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY is not set")

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                _TOGETHER_CHAT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": _DEFAULT_LLM, "messages": messages},
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            from services.llm import _token_counter
            _token_counter.record(tokens)
            return LLMResponse(
                text=data["choices"][0]["message"]["content"],
                tokens_used=tokens,
            )
