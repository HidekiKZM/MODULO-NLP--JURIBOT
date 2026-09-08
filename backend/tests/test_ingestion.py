"""Deterministic integration tests against an isolated in-memory Qdrant."""
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from qdrant_client import QdrantClient, models

from backend.core.config import Settings
from backend.core.runtime import Runtime, SearchUnavailable
from backend.ingest.prepare_index import (
    Document, DocUnit, PDFLoader, clean_legal_text, document_id, document_version,
    embedding_batches, ingest_documents, make_splitter,
)
from backend.rag.index_contract import IncompatibleIndex, MANIFEST_ID, active_filter, search_index
from backend.rag.vector_qdrant import QdrantManager


class Encoder:
    model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 3)

    def __init__(self):
        self.batch_sizes = []

    def generate_batch(self, texts):
        self.batch_sizes.append(len(texts))
        return [[1.0, 0.1, 0.1] if "multa" in text.lower() else [0.1, 1.0, 0.1] for text in texts]


def document(text, doc_id="doc-a", source_hash="bytes-v1"):
    return Document(doc_id, source_hash, [DocUnit(doc_id, "pdf", "Contrato", "contrato.pdf", 1, text, {})])


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, QDRANT_COLLECTION="validation-new", CHUNK_SIZE=80,
                                 CHUNK_OVERLAP=0, EMBEDDING_BATCH_SIZE=2, QDRANT_BATCH_SIZE=2)
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        self.manager = QdrantManager(settings=self.settings, client=self.client)
        self.encoder = Encoder()
        self.manager.ensure_collection_exists(3)

    def publish(self, doc):
        return self.manager.publish_document(doc.doc_id, document_version(doc, self.settings),
            embedding_batches(doc, self.encoder, self.settings), 3, doc.source_hash)

    def chunks(self):
        condition = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))])
        return self.client.scroll(self.settings.QDRANT_COLLECTION, scroll_filter=condition, limit=100)[0]

    def test_values_search_and_repeat_do_not_duplicate(self):
        doc = document("Art. 1 Multa de 10% sobre R$ 1.250,00; valor >= R$ 50,00.")
        self.publish(doc)
        before = {point.id for point in self.chunks()}
        self.publish(doc)
        self.assertEqual(before, {point.id for point in self.chunks()})
        hits = search_index(self.client, self.settings.QDRANT_COLLECTION, self.settings, [1., .1, .1], 5)
        self.assertEqual(len(hits), 1)
        self.assertIn("10%", hits[0].payload["content"])
        self.assertIn("R$ 1.250,00", hits[0].payload["content"])

    def test_update_removes_obsolete_chunks_only_for_updated_document(self):
        self.publish(document("Art. 1 Multa de 10%.\n" * 20))
        self.publish(document("Outro documento preservado.", "doc-b"))
        old_count = len(self.chunks())
        replacement = document("Multa atual de 2%.", source_hash="bytes-v2")
        self.publish(replacement)
        chunks = self.chunks()
        self.assertLess(len(chunks), old_count)
        self.assertEqual(len(chunks), 2)
        self.assertEqual({p.payload["doc_id"] for p in chunks}, {"doc-a", "doc-b"})
        self.assertTrue(all(p.payload["version"] == document_version(replacement, self.settings)
                            for p in chunks if p.payload["doc_id"] == "doc-a"))

    def test_failed_batch_does_not_publish_new_version_and_retry_cleans_staging(self):
        self.publish(document("Multa anterior de 10%."))
        replacement = document("Multa nova de 2%.", source_hash="v2")
        version = document_version(replacement, self.settings)
        def failed_batches():
            yield [[1., .1, .1]], [{"content": "PARCIAL"}]
            raise OSError("simulated interruption")
        with self.assertRaises(OSError):
            self.manager.publish_document("doc-a", version, failed_batches(), 3, "v2")
        hits = search_index(self.client, self.settings.QDRANT_COLLECTION, self.settings, [1., .1, .1], 5)
        self.assertEqual([p.payload["content"] for p in hits], ["Multa anterior de 10%."])
        self.publish(replacement)
        self.assertEqual(len(self.chunks()), 1)

    def test_batches_are_bounded(self):
        self.publish(document("Art. 1 Multa de 10% e prazo de 30 dias.\n" * 25))
        self.assertGreater(len(self.encoder.batch_sizes), 1)
        self.assertLessEqual(max(self.encoder.batch_sizes), 2)

    def test_existing_legacy_collection_is_untouched(self):
        self.client.create_collection("legacy", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
        self.client.upsert("legacy", [models.PointStruct(id=1, vector=[1., .1, .1], payload={"old": True})])
        manager = QdrantManager("legacy", settings=self.settings, client=self.client)
        with self.assertRaises(IncompatibleIndex):
            manager.ensure_collection_exists(3)
        self.assertEqual(self.client.retrieve("legacy", [1])[0].payload, {"old": True})
        self.assertEqual(self.client.count("legacy").count, 1)

    def test_model_revision_dimension_and_metric_mismatch_are_rejected(self):
        for changes in [{"EMBEDDING_MODEL": "other/model"}, {"EMBEDDING_REVISION": "a" * 40}]:
            config = self.settings.model_copy(update=changes)
            manager = QdrantManager(settings=config, client=self.client)
            with self.subTest(changes=changes), self.assertRaises(IncompatibleIndex):
                manager.ensure_collection_exists(3)
            with self.assertRaises(IncompatibleIndex):
                search_index(self.client, config.QDRANT_COLLECTION, config, [1., .1, .1], 5)
        with self.assertRaises(IncompatibleIndex):
            self.manager.ensure_collection_exists(4)
        self.client.create_collection("wrong-metric", vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT))
        with self.assertRaises(IncompatibleIndex):
            QdrantManager("wrong-metric", settings=self.settings, client=self.client).ensure_collection_exists(3)

    def test_empty_invalid_vectors_and_manifest_only_index_are_not_success(self):
        self.assertIsNone(active_filter(self.client, self.settings.QDRANT_COLLECTION))
        for batches in [[], [([[1., 2.]], [{}])], [([[float("nan"), 1., 1.]], [{}])], [([[0., 0., 0.]], [{}])]]:
            with self.assertRaises(ValueError):
                self.manager.publish_document("doc", "v1", iter(batches), 3, "hash")
        self.assertIsNone(active_filter(self.client, self.settings.QDRANT_COLLECTION))

    def test_transport_failure_never_creates_collection(self):
        with patch.object(self.client, "collection_exists", side_effect=ConnectionError), \
             patch.object(self.client, "create_collection") as create:
            with self.assertRaises(ConnectionError):
                self.manager.ensure_collection_exists(3)
            create.assert_not_called()

    def test_runtime_rejects_unknown_model_before_encoding(self):
        self.client.delete(self.settings.QDRANT_COLLECTION, models.PointIdsList(points=[MANIFEST_ID]))
        model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 3)
        runtime = Runtime(model=model, qdrant=self.client, probe=self.client)
        self.assertEqual(runtime.readiness(self.settings)[1]["collection"], "incompatible")
        with self.assertRaises(SearchUnavailable):
            runtime.search(self.settings, "teste", 5)


class TextAndExtractionTests(unittest.TestCase):
    def test_model_loading_pins_revision_and_disables_remote_code(self):
        from backend.rag.embeddings import load_embedding_model
        config = Settings(_env_file=None)
        constructor, download = Mock(), Mock()
        with patch.dict(sys.modules, {
            "sentence_transformers": SimpleNamespace(SentenceTransformer=constructor),
            "huggingface_hub": SimpleNamespace(hf_hub_download=download),
        }):
            load_embedding_model(config)
        download.assert_called_once_with(config.EMBEDDING_MODEL, "modules.json", revision=config.EMBEDDING_REVISION)
        constructor.assert_called_once_with(config.EMBEDDING_MODEL, revision=config.EMBEDDING_REVISION,
                                             device="cpu", trust_remote_code=False)

    def test_mutable_model_revision_is_rejected(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            Settings(_env_file=None, EMBEDDING_REVISION="main")

    def test_cleaning_preserves_financial_and_math_symbols(self):
        text = "Art. 1\x00 Multa: 10%; R$ 1.250,00; x >= y; x ≤ 5; + − × ÷ = § 2º."
        cleaned = clean_legal_text(text)
        for symbol in ["10%", "R$ 1.250,00", ">=", "≤", "+", "−", "×", "÷", "=", "§ 2º"]:
            self.assertIn(symbol, cleaned)
        self.assertNotIn("\x00", cleaned)

    def test_heading_regex_preserves_headings_and_punctuation(self):
        text = "CAPÍTULO I\nArt. 1 " + "multa de 10%. " * 5 + "\nArt. 2 " + "R$ 50,00. " * 5
        chunks = make_splitter(100, 0).split_text(text)
        self.assertTrue(any(chunk.startswith("Art. 2") for chunk in chunks))
        self.assertIn("CAPÍTULO I", chunks[0])
        self.assertTrue(any("10%." in chunk for chunk in chunks))

    def test_identity_is_relative_and_version_tracks_content_and_chunking(self):
        from backend.tests.smoke_ingestion import write_sample_pdf
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_pdf, second_pdf = Path(first, "cdc.pdf"), Path(second, "cdc.pdf")
            write_sample_pdf(first_pdf, ["Art. 1 Multa de 10% sobre R$ 100,00."])
            write_sample_pdf(second_pdf, ["Art. 1 Multa de 2% sobre R$ 50,00."])
            original_doc = PDFLoader(first, ocr=False).load_document(first_pdf)
            relocated_doc = PDFLoader(second, ocr=False).load_document(second_pdf)
            self.assertEqual(original_doc.doc_id, relocated_doc.doc_id)
            self.assertNotEqual(original_doc.source_hash, relocated_doc.source_hash)
        self.assertNotEqual(document_id(Path("leis/cdc.pdf"), "juribot"), document_id(Path("leis/cdc.pdf"), "other"))
        config = Settings(_env_file=None)
        original = document("Valor de R$ 50,00")
        changed = document("Valor de R$ 100,00", source_hash="v2")
        self.assertNotEqual(document_version(original, config), document_version(changed, config))
        self.assertNotEqual(document_version(original, config), document_version(original, config.model_copy(update={"CHUNK_SIZE": 900})))

    def test_empty_directory_and_corrupt_pdf_fail_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            loader = PDFLoader(directory)
            with self.assertRaisesRegex(ValueError, "No PDF"):
                list(loader.iter_documents())
            Path(directory, "bad.pdf").write_bytes(b"not a PDF")
            self.assertEqual(list(loader.iter_documents()), [])
            self.assertEqual(len(loader.failures), 1)

    def test_extraction_and_ocr_failure_reject_document(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "sample.pdf")
            path.write_bytes(b"placeholder")
            loader = PDFLoader(directory)
            fake_pdf = MagicMock()
            fake_pdf.__enter__.return_value.pages = [SimpleNamespace(extract_text=lambda: "")]
            with patch("backend.ingest.prepare_index.pdfplumber.open", return_value=fake_pdf), \
                 patch.object(loader, "_ocr_page", side_effect=OSError("no OCR")):
                self.assertEqual(list(loader.iter_documents()), [])
            self.assertIn("OCR failed", loader.failures[0]["error"])

    def test_zero_documents_never_returns_success(self):
        loader = SimpleNamespace(iter_documents=lambda **kwargs: iter([]), failures=[])
        manager = SimpleNamespace(ensure_collection_exists=lambda **kwargs: None)
        with self.assertRaisesRegex(RuntimeError, "Ingestion incomplete"):
            ingest_documents(loader, manager, Encoder(), Settings(_env_file=None))


if __name__ == "__main__":
    unittest.main()
