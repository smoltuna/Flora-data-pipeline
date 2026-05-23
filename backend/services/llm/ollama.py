import httpx
from config import settings


class OllamaProvider:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.llm_model = settings.ollama_llm_model
        self.embed_model = settings.ollama_embed_model

    async def complete(self, prompt: str, system: str = "") -> str:
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
            from services.llm import _token_counter
            _token_counter.record(
                data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
            )
            return data["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]
