# rag/vector_qdrant.py
import uuid
from backend.core.config import get_settings
from backend.core.runtime import create_qdrant_client
from qdrant_client.models import Distance, VectorParams

class QdrantManager:
    def __init__(self, collection_name: str | None = None):
        self.settings = get_settings()
        self.collection = collection_name or self.settings.QDRANT_COLLECTION
        self.client = create_qdrant_client(self.settings)

    def ensure_collection_exists(self, vector_size: int = 384):
        try:
            self.client.get_collection(self.collection)
            print(f"INFO: Coleção '{self.collection}' já existe.")
        except Exception:
            print(f"INFO: Coleção '{self.collection}' não encontrada. Criando...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print("INFO: Coleção criada com sucesso.")

    def upsert_points(self, vectors, payloads):
        batch_size = self.settings.QDRANT_BATCH_SIZE
        parallel = self.settings.QDRANT_PARALLEL

        # ✅ IDs estáveis como UUID v5 (válidos para o Qdrant)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{p.get('doc_id','id')}:{p.get('chunk_no', i)}"))
            for i, p in enumerate(payloads)
        ]

        self.client.upload_collection(
            collection_name=self.collection,
            vectors=vectors,
            payload=payloads,
            ids=ids,                 # <- agora são UUIDs válidos
            batch_size=batch_size,
            parallel=parallel,
            max_retries=3,
        )
        print("✅ Upload para Qdrant concluído.")
