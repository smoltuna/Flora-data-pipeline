import httpx
from config import settings


class OpenAIEmbedding:
    """OpenAI embedding provider using text-embedding-3-small (1536-dim)."""

    def __init__(self) -> None:
        self.api_key = settings.openai_api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.model = getattr(settings, "openai_embed_model", "text-embedding-3-small")

    @property
    def dimensions(self) -> int:
        return 1536

    async def embed(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()["data"]
            # OpenAI returns data sorted by index
            data.sort(key=lambda x: x["index"])
            return [d["embedding"] for d in data]
