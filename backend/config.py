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

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "flora-assets"

    # fal.ai (vision judge + FLUX lock icon generation)
    fal_key: str = ""

    # Web search (5th source — DuckDuckGo + httpx/bs4)
    web_search_enabled: bool = True

    # Chunking config (Session 3)
    chunk_size: int = 500
    chunk_overlap: int = 50

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"

    # App
    log_level: str = "INFO"
    environment: str = "development"


settings = Settings()
