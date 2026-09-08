"""Run from the repository root: python -m backend.scripts.smoke_qdrant."""
from backend.core.config import get_settings
from backend.core.runtime import create_qdrant_client
from backend.rag.embeddings import load_embedding_model
from backend.rag.index_contract import search_index


def main():
    settings = get_settings()
    model = load_embedding_model(settings)
    vector = model.encode("prazo de arrependimento em compra online", normalize_embeddings=True).tolist()
    client = create_qdrant_client(settings)
    try:
        hits = search_index(client, settings.QDRANT_COLLECTION, settings, vector, 5)
        for index, hit in enumerate(hits, 1):
            payload = hit.payload or {}
            print(index, round(hit.score, 4), payload.get("title"), payload.get("page"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
