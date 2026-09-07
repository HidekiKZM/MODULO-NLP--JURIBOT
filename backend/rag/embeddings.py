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
    return SentenceTransformer(settings.EMBEDDING_MODEL, device=device)


@lru_cache(maxsize=1)
def get_embedding_model() -> "SentenceTransformer":
    return load_embedding_model(get_settings())


class EmbeddingGenerator:
    def __init__(self):
        self.model = get_embedding_model()

    def generate(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()
