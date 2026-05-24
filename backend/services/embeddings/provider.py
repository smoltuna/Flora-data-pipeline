from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...


def get_embedding_provider(provider_name: str | None = None) -> EmbeddingProvider:
    """Return an embedding provider instance based on config."""
    from config import settings

    name = provider_name or settings.embed_provider

    if name == "openai":
        from services.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding()
    else:
        from services.embeddings.ollama import OllamaEmbedding
        return OllamaEmbedding()
