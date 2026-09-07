"""Run from the repository root: python -m backend.tools.qdrant_check."""
from backend.core.config import get_settings
from backend.core.runtime import create_qdrant_client


def main():
    settings = get_settings()
    client = create_qdrant_client(settings, probe=True)
    try:
        info = client.get_collection(settings.QDRANT_COLLECTION)
        vectors = info.config.params.vectors
        print(f"Collection={settings.QDRANT_COLLECTION} | metric={vectors.distance.value} | dim={vectors.size}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
