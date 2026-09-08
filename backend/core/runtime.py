"""Application-owned resources; importing this module does not load ML models."""
from dataclasses import dataclass
import logging
from typing import Any

from backend.core.config import Settings
from backend.rag.index_contract import IncompatibleIndex, active_filter, search_index, validate_index

logger = logging.getLogger(__name__)


def create_qdrant_client(settings: Settings, *, probe: bool = False):
    from qdrant_client import QdrantClient

    address = {"url": settings.QDRANT_URL} if settings.QDRANT_URL else {"host": settings.QDRANT_HOST}
    return QdrantClient(
        **address,
        port=None if settings.QDRANT_URL else settings.QDRANT_PORT,
        grpc_port=settings.QDRANT_GRPC_PORT,
        prefer_grpc=settings.QDRANT_USE_GRPC,
        api_key=settings.QDRANT_API_KEY.get_secret_value() or None,
        timeout=settings.READINESS_TIMEOUT if probe else settings.QDRANT_TIMEOUT,
    )


def create_gemini_model(settings: Settings):
    key, model_name = settings.require_gemini()
    import google.generativeai as genai

    genai.configure(api_key=key)
    return genai.GenerativeModel(model_name)


class SearchUnavailable(RuntimeError):
    pass


@dataclass
class Runtime:
    model: Any = None
    qdrant: Any = None
    probe: Any = None
    gemini: Any = None

    def readiness(self, settings: Settings) -> tuple[bool, dict[str, str]]:
        checks = {
            "embeddings": "ready" if self.model is not None else "unavailable",
            "qdrant": "unavailable",
            "collection": "unavailable",
            # Configuration check, not a paid call to Google.
            "gemini": "disabled" if not settings.gemini_enabled else (
                "configured" if self.gemini is not None else "unavailable"
            ),
        }
        if self.probe is not None:
            try:
                info = self.probe.get_collection(settings.QDRANT_COLLECTION)
                checks["qdrant"] = "ready" if self.qdrant is not None else "unavailable"
                dimension = self.model.get_sentence_embedding_dimension() if self.model is not None else None
                validate_index(self.probe, settings.QDRANT_COLLECTION, settings, dimension, info=info)
                checks["collection"] = "ready" if active_filter(self.probe, settings.QDRANT_COLLECTION) else "empty"
            except IncompatibleIndex:
                checks["collection"] = "incompatible"
            except Exception as error:
                code = error.code() if callable(getattr(error, "code", None)) else None
                if getattr(error, "status_code", None) == 404 or getattr(code, "name", None) == "NOT_FOUND":
                    checks["qdrant"] = "ready" if self.qdrant is not None else "unavailable"
                    checks["collection"] = "missing"
                # No exception strings, URLs or credentials in health responses.
        ready = all(checks[name] == "ready" for name in ("embeddings", "qdrant", "collection"))
        return ready and checks["gemini"] != "unavailable", checks

    def search(self, settings: Settings, query: str, top_k: int) -> list[dict]:
        if self.model is None or self.qdrant is None:
            raise SearchUnavailable("Recursos de busca indisponíveis; consulte /ready")
        try:
            dimension = self.model.get_sentence_embedding_dimension()
            validate_index(self.qdrant, settings.QDRANT_COLLECTION, settings, dimension)
            hits = search_index(self.qdrant, settings.QDRANT_COLLECTION, settings,
                                self.model.encode(query, normalize_embeddings=True).tolist(), top_k)
            return [
                {
                    "score": float(hit.score),
                    "title": (hit.payload or {}).get("title"),
                    "page": (hit.payload or {}).get("page"),
                    "uri": (hit.payload or {}).get("uri"),
                    "snippet": ((hit.payload or {}).get("content") or "")[:400],
                }
                for hit in hits
            ]
        except Exception as error:
            logger.warning("Busca indisponível (%s)", type(error).__name__)
            raise SearchUnavailable("Busca indisponível; consulte /ready") from error

    def close(self):
        for client in (self.qdrant, self.probe):
            if client is not None:
                try:
                    client.close()
                except Exception as error:
                    logger.warning("Falha ao fechar cliente (%s)", type(error).__name__)
        self.model = self.qdrant = self.probe = self.gemini = None


def load_runtime(settings: Settings) -> Runtime:
    from backend.rag.embeddings import load_embedding_model

    runtime = Runtime()
    factories = {
        "model": lambda: load_embedding_model(settings),
        "qdrant": lambda: create_qdrant_client(settings),
        "probe": lambda: create_qdrant_client(settings, probe=True),
    }
    if settings.gemini_enabled:
        factories["gemini"] = lambda: create_gemini_model(settings)
    for name, factory in factories.items():
        try:
            setattr(runtime, name, factory())
        except Exception as error:
            # Resource outages produce 503 readiness, never missing routes.
            logger.error("Inicialização de %s falhou (%s)", name, type(error).__name__)
    return runtime
