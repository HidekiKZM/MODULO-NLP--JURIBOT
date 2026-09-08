"""Safe creation and per-document publication of a versioned index."""
import uuid

from qdrant_client import models

from backend.core.config import get_settings
from backend.rag.index_contract import (
    MANIFEST_ID, identity, match, search_index, validate_index, validate_vectors,
)


def document_point_id(doc_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "juribot:document:" + doc_id))


class QdrantManager:
    def __init__(self, collection_name=None, *, settings=None, client=None):
        from backend.core.runtime import create_qdrant_client
        self.settings = settings or get_settings()
        self.collection = collection_name or self.settings.QDRANT_COLLECTION
        self.client = client if client is not None else create_qdrant_client(self.settings)

    def ensure_collection_exists(self, vector_size: int):
        if vector_size <= 0:
            raise ValueError("Invalid embedding dimension")
        if not self.client.collection_exists(self.collection):
            # Never recreate/delete a collection. A concurrent creator causes an error.
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            self.client.upsert(self.collection, points=[models.PointStruct(
                id=MANIFEST_ID, vector=[0.0] * vector_size,
                payload=identity(self.settings, vector_size),
            )], wait=True)
        validate_index(self.client, self.collection, self.settings, vector_size)

    def publish_document(self, doc_id, version, batches, dimension, source_hash):
        """Only publish after every batch succeeds, then remove obsolete versions.

        Requires one ingestion writer per collection. Repeated versions use the same
        point IDs. A failed upload leaves unpublished chunks for the next retry;
        it does not replace the previously published document.
        """
        validate_index(self.client, self.collection, self.settings, dimension)
        count = 0
        for vectors, payloads in batches:
            validate_vectors(vectors, dimension)
            if len(vectors) != len(payloads):
                raise ValueError("Embedding/payload count mismatch")
            points = []
            for vector, payload in zip(vectors, payloads):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{version}:{count}"))
                points.append(models.PointStruct(id=point_id, vector=vector, payload={
                    **payload, "kind": "chunk", "doc_id": doc_id,
                    "version": version, "chunk_no": count,
                }))
                count += 1
            self.client.upsert(self.collection, points=points, wait=True)
        if count == 0:
            raise ValueError("Document contains no indexable chunks")
        self.client.upsert(self.collection, points=[models.PointStruct(
            id=document_point_id(doc_id), vector=[0.0] * dimension,
            payload={"kind": "document", "doc_id": doc_id, "version": version,
                     "source_hash": source_hash, "chunks": count},
        )], wait=True)
        # Restrict cleanup to obsolete chunks of this successfully published document.
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=models.Filter(
            must=[match("kind", "chunk"), match("doc_id", doc_id)],
            must_not=[match("version", version)],
        )), wait=True)
        return count

    def search(self, query_vector, limit=5):
        hits = search_index(self.client, self.collection, self.settings, query_vector, limit)
        return [{"score": hit.score, "payload": hit.payload} for hit in hits]

    def close(self):
        self.client.close()
