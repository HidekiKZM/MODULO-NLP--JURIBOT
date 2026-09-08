"""Embedding configuration shared by API startup and ingestion."""
from functools import lru_cache
from typing import TYPE_CHECKING

from backend.core.config import Settings, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def load_embedding_model(settings: Settings) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    device = settings.EMBEDDING_DEVICE
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from pathlib import Path
    if Path(settings.EMBEDDING_MODEL).exists():
        raise ValueError("Use a Hugging Face model ID and immutable revision, not a mutable local directory")
    from huggingface_hub import hf_hub_download
    # Fail instead of silently constructing a different pooling pipeline on download errors.
    hf_hub_download(settings.EMBEDDING_MODEL, "modules.json", revision=settings.EMBEDDING_REVISION)
    return SentenceTransformer(
        settings.EMBEDDING_MODEL, revision=settings.EMBEDDING_REVISION,
        device=device, trust_remote_code=False,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformer":
    return load_embedding_model(get_settings())


class EmbeddingGenerator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = load_embedding_model(self.settings) if settings is not None else get_embedding_model()

    def generate(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_tensor=False, normalize_embeddings=True).tolist()

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True,
                                 batch_size=self.settings.EMBEDDING_BATCH_SIZE).tolist()
