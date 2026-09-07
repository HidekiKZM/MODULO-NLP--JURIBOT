"""Configuration shared by the API, ingestion and diagnostic commands."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
UI_DIR = BACKEND_DIR / "ui"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    ENV: str = "development"
    GEMINI_API_KEY: SecretStr = SecretStr("")
    # Generation is optional; do not select a retired model implicitly.
    GEMINI_MODEL: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    CHUNK_SIZE: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP: int = Field(default=150, ge=0)
    DATA_DIR: Path = PROJECT_ROOT / "data"
    POPPLER_PATH: str | None = None

    QDRANT_URL: str | None = None
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = Field(default=6333, ge=1, le=65535)
    QDRANT_GRPC_PORT: int = Field(default=6334, ge=1, le=65535)
    QDRANT_USE_GRPC: bool = True
    QDRANT_API_KEY: SecretStr = SecretStr("")
    QDRANT_COLLECTION: str = "juribot_chunks"
    QDRANT_TIMEOUT: float = Field(default=30, gt=0)
    READINESS_TIMEOUT: float = Field(default=2, gt=0, le=10)
    QDRANT_BATCH_SIZE: int = Field(default=256, gt=0)
    QDRANT_PARALLEL: int = Field(default=2, gt=0)

    # Optional components do not affect readiness until integrated into the API.
    REDIS_URL: str = "redis://localhost:6379"
    USE_OPENSEARCH: bool = False
    OPENSEARCH_URL: str = "http://localhost:9200"
    ENABLE_RERANKER: bool = False
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    JWT_SECRET_KEY: SecretStr = SecretStr("")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    RATE_LIMIT_PER_MINUTE: int = 30

    @field_validator("DATA_DIR", mode="before")
    @classmethod
    def resolve_data_dir(cls, value):
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @field_validator("QDRANT_URL", "POPPLER_PATH", mode="before")
    @classmethod
    def empty_to_none(cls, value):
        return (value.strip() or None) if isinstance(value, str) else value

    @field_validator("GEMINI_MODEL", "EMBEDDING_MODEL", "EMBEDDING_DEVICE", "QDRANT_HOST", "QDRANT_COLLECTION")
    @classmethod
    def strip_text(cls, value):
        return value.strip()

    @model_validator(mode="after")
    def validate_chunking(self):
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP deve ser menor que CHUNK_SIZE")
        if not self.EMBEDDING_MODEL or not self.QDRANT_COLLECTION or not self.QDRANT_HOST:
            raise ValueError("Modelo de embeddings, host e coleção não podem ser vazios")
        return self

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.GEMINI_API_KEY.get_secret_value().strip())

    def require_gemini(self) -> tuple[str, str]:
        """Only generation calls this; indexing never needs a Gemini key."""
        if not self.gemini_enabled:
            raise ValueError("GEMINI_API_KEY é necessária para geração com Gemini")
        if not self.GEMINI_MODEL:
            raise ValueError("GEMINI_MODEL é necessário para geração com Gemini")
        return self.GEMINI_API_KEY.get_secret_value().strip(), self.GEMINI_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
