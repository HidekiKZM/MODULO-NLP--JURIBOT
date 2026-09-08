"""Manual integration: real pinned embeddings, generated PDF, new Qdrant collection.

Requires a disposable Qdrant server. Never points at the project's data directory.
No collection is dropped; the randomly named collection remains for inspection.
"""
import json
from pathlib import Path
import tempfile
import uuid

from backend.core.config import Settings
from backend.core.runtime import Runtime, create_qdrant_client
from backend.ingest.prepare_index import PDFLoader, ingest_documents
from backend.rag.embeddings import EmbeddingGenerator
from backend.rag.vector_qdrant import QdrantManager


def write_sample_pdf(path, lines):
    """Tiny text PDF fixture, generated without extra production dependencies."""
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    stream = ("BT /F1 12 Tf 50 780 Td 18 TL " + " T* ".join(f"({line}) Tj" for line in escaped) + " ET").encode("latin-1")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(content)
    content.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    path.write_bytes(content)


def main():
    collection = "juribot_validation_" + uuid.uuid4().hex
    settings = Settings(_env_file=None, QDRANT_COLLECTION=collection,
                        CHUNK_SIZE=180, CHUNK_OVERLAP=20, EMBEDDING_BATCH_SIZE=2)
    client = create_qdrant_client(settings)
    assert not client.collection_exists(collection)
    encoder = EmbeddingGenerator(settings)
    manager = QdrantManager(settings=settings, client=client)
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "contrato.pdf")
            write_sample_pdf(path, ["CAPITULO I - VALORES", "Art. 1 Multa de 10% sobre R$ 1.250,00.",
                                    "Art. 2 Prazo de arrependimento de 7 dias para compra online."])
            loader = lambda: PDFLoader(directory, ocr=False, namespace="integration-fixture")
            report = ingest_documents(loader(), manager, encoder, settings)
            count = client.count(collection, exact=True).count
            repeated = ingest_documents(loader(), manager, encoder, settings)
            assert client.count(collection, exact=True).count == count
            runtime = Runtime(model=encoder.model, qdrant=client, probe=client)
            hits = runtime.search(settings, "Qual o percentual da multa e o valor em reais?", 3)
            assert any("10%" in hit["snippet"] and "R$ 1.250,00" in hit["snippet"] for hit in hits)
            assert runtime.readiness(settings)[0]
            write_sample_pdf(path, ["Art. 1 Multa atualizada de 2% sobre R$ 500,00."])
            updated = ingest_documents(loader(), manager, encoder, settings)
            hits = runtime.search(settings, "Qual o percentual da multa atualizada?", 3)
            assert any("2%" in hit["snippet"] and "R$ 500,00" in hit["snippet"] for hit in hits)
            assert all("10%" not in hit["snippet"] for hit in hits)
            print(json.dumps({"collection": collection, "model": settings.EMBEDDING_MODEL,
                "revision": settings.EMBEDDING_REVISION, "dimension": encoder.model.get_sentence_embedding_dimension(),
                "first": report, "repeat": repeated, "update": updated,
                "points_after_repeat": count, "points_after_update": client.count(collection, exact=True).count,
                "search": hits, "result": "PASS"}, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
