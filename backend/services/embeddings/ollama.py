import httpx
from config import settings


class OllamaEmbedding:
    """Ollama embedding provider using nomic-embed-text (768-dim)."""

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_embed_model

    @property
    def dimensions(self) -> int:
        return 768

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed using Ollama's /api/embed endpoint."""
        if not texts:
            return []
        if len(texts) == 1:
            return [await self.embed(texts[0])]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts},
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]
        except Exception:
            # Fall back to sequential if batch endpoint unavailable
            return [await self.embed(t) for t in texts]
