"""Versioned index contract for Qdrant 1.9 (reserved non-searchable records)."""
import math
import uuid

from qdrant_client import models

from backend.core.config import Settings

MANIFEST_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "juribot:index-contract:v2"))


class IncompatibleIndex(ValueError):
    pass


def identity(settings: Settings, dimension: int) -> dict:
    return {
        "kind": "index_manifest", "schema": 2,
        "model": settings.EMBEDDING_MODEL, "revision": settings.EMBEDDING_REVISION,
        "dimension": dimension, "distance": "Cosine", "normalize": True,
        "embedding_pipeline": "sentence-transformers-v1",
    }


def match(key, value):
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def validate_index(client, collection: str, settings: Settings, dimension: int, info=None):
    info = info or client.get_collection(collection)
    vectors = info.config.params.vectors
    distance = getattr(vectors, "distance", None)
    if (isinstance(vectors, dict) or getattr(vectors, "size", None) != dimension
            or getattr(distance, "value", distance) != "Cosine"):
        raise IncompatibleIndex("Index dimension or distance differs; use a new collection")
    records = client.retrieve(collection, ids=[MANIFEST_ID], with_payload=True)
    if len(records) != 1 or records[0].payload != identity(settings, dimension):
        raise IncompatibleIndex("Unknown or incompatible embedding model/revision; use a new collection")


def validate_vectors(vectors, dimension):
    if not vectors:
        raise ValueError("Empty embeddings batch")
    for vector in vectors:
        if len(vector) != dimension or not all(math.isfinite(float(v)) for v in vector):
            raise ValueError("Invalid embedding dimension or non-finite values")
        if not any(vector):
            raise ValueError("Zero embedding")


def active_filter(client, collection):
    """Snapshot published document versions; staged/obsolete chunks stay invisible."""
    versions = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection, scroll_filter=models.Filter(must=[match("kind", "document")]),
            limit=256, offset=offset, with_payload=True, with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            if not payload.get("doc_id") or not payload.get("version"):
                raise IncompatibleIndex("Invalid document manifest")
            versions.append(models.Filter(must=[match("doc_id", payload["doc_id"]),
                                                match("version", payload["version"])]))
        if offset is None:
            break
    if not versions:
        return None
    return models.Filter(must=[match("kind", "chunk")], should=versions)


def search_index(client, collection, settings, vector, limit):
    validate_vectors([vector], len(vector))
    validate_index(client, collection, settings, len(vector))
    query_filter = active_filter(client, collection)
    if query_filter is None:
        return []
    return client.search(collection_name=collection, query_vector=vector,
                         query_filter=query_filter, limit=limit, with_payload=True)
