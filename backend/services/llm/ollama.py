import httpx
from config import settings

from services.llm.provider import LLMResponse


class OllamaProvider:
    def __init__(self, model_override: str | None = None) -> None:
        self.base_url = settings.ollama_base_url
        self.llm_model = model_override or settings.ollama_llm_model

    async def complete(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.llm_model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
            tokens = data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            from services.llm import _token_counter
            _token_counter.record(tokens)
            return LLMResponse(
                text=data["message"]["content"],
                tokens_used=tokens,
            )
