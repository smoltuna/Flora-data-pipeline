from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://flora:flora@localhost:5432/flora"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # Cloud LLM providers
    gemini_api_key: str = ""
    groq_api_key: str = ""
    together_api_key: str = ""
    openai_api_key: str = ""

    llm_provider: str = "ollama"  # ollama | gemini | groq | together | openai

    # Separate provider for translation — defaults to ollama (no rate limits, runs locally).
    # Set TRANSLATION_PROVIDER=groq in .env only if you have a paid Groq account with
    # sufficient TPM quota (free tier ~14,400 TPM is too low for batch translation).
    translation_provider: str = "ollama"

    # Per-step provider overrides (Session 4)
    embed_provider: str = "ollama"          # ollama | openai
    grade_provider: str = ""                # falls back to llm_provider if empty
    query_gen_provider: str = ""            # falls back to llm_provider if empty
    judge_provider: str = ""                # falls back to llm_provider if empty

    # Per-step Ollama model overrides
    grade_model: str = ""                   # e.g. "llama3.2:1b" — falls back to ollama_llm_model
    query_gen_model: str = ""               # e.g. "llama3.2:3b"
    synth_model: str = ""                   # e.g. "qwen2.5:7b" — recommended for fewer hallucinations
    judge_model: str = ""                   # e.g. "llama3.2:3b"
    translation_model: str = ""             # e.g. "qwen2.5:7b" — strong multilingual on M4 Metal
    fact_check_model: str = ""              # e.g. "qwen2.5:7b" — used to compare claim vs web snippet

    # Post-synthesis web fact-check for COMPLEX fields (etymology, cultural_info, fun_fact).
    # Runs a targeted DDG search per field, compares the synthesized claim to the top snippet,
    # regenerates the field if they disagree. Bounds the LLM cost by capping to one regen.
    fact_check_enabled: bool = True

    # OpenAI embeddings (when EMBED_PROVIDER=openai)
    openai_embed_model: str = "text-embedding-3-small"

    # fal.ai (vision judge + FLUX lock icon generation)
    fal_key: str = ""

    # Web search (5th source — DuckDuckGo + httpx/bs4)
    web_search_enabled: bool = True

    # Chunking config (Session 3)
    chunk_size: int = 500
    chunk_overlap: int = 50

    # LLM-as-Judge (Session 8) — fields scoring below this on factual_accuracy
    # are logged as warnings (not blocked).
    quality_gate_threshold: float = 0.5

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"

    # App
    log_level: str = "INFO"
    environment: str = "development"


settings = Settings()
