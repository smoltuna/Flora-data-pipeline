from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LLMResponse:
    text: str
    tokens_used: int = 0  # prompt + completion


@runtime_checkable
class LLMProvider(Protocol):
    async def complete(self, prompt: str, system: str = "") -> LLMResponse: ...


def get_provider(
    provider_name: str | None = None,
    *,
    step: str | None = None,
) -> LLMProvider:
    """Return an LLM provider instance.

    If *step* is given (e.g. "grade", "query_gen", "judge", "synth",
    "translation", "fact_check"), the per-step provider and model settings
    are used when available, falling back to the default LLM_PROVIDER /
    OLLAMA_LLM_MODEL.
    """
    from config import settings

    # Resolve provider name from step-specific config if available.
    if step and not provider_name:
        step_provider_attr = f"{step}_provider"
        provider_name = getattr(settings, step_provider_attr, None)

    name = provider_name or settings.llm_provider

    # Resolve model override for Ollama steps.
    step_model: str | None = None
    if step:
        step_model_attr = f"{step}_model"
        step_model = getattr(settings, step_model_attr, None) or None

    if name == "gemini":
        from services.llm.gemini import GeminiProvider
        return GeminiProvider()
    elif name == "groq":
        from services.llm.groq import GroqProvider
        return GroqProvider()
    elif name == "together":
        from services.llm.together import TogetherProvider
        return TogetherProvider()
    else:
        from services.llm.ollama import OllamaProvider
        return OllamaProvider(model_override=step_model)
