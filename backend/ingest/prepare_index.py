"""PDF ingestion with immutable document versions and bounded embedding batches."""
import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import re
import tempfile

import ftfy
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import portalocker
import pytesseract

from backend.core.config import get_settings
from backend.rag.embeddings import EmbeddingGenerator
from backend.rag.vector_qdrant import QdrantManager

logger = logging.getLogger(__name__)
PIPELINE_VERSION = "legal-text-v2"


@dataclass
class DocUnit:
    doc_id: str
    source: str
    title: str
    uri: str
    page: int
    text: str
    extra: dict


@dataclass
class Document:
    doc_id: str
    source_hash: str
    units: list[DocUnit]


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document_id(relative_path, namespace):
    # Independent of machine, checkout directory, file contents and page count.
    return hashlib.sha256(f"{namespace}:{relative_path.as_posix()}".encode("utf-8")).hexdigest()


class PDFLoader:
    def __init__(self, data_dir, ocr=True, poppler_path=None, namespace="juribot"):
        self.data_dir = Path(data_dir).resolve()
        self.ocr = ocr
        self.poppler_path = poppler_path
        self.namespace = namespace
        self.failures = []

    def _ocr_page(self, path, page_num):
        images = convert_from_path(str(path), first_page=page_num + 1, last_page=page_num + 1,
                                   poppler_path=self.poppler_path, timeout=120)
        if not images:
            raise ValueError("OCR conversion returned no image")
        try:
            return pytesseract.image_to_string(images[0], lang="por+eng", timeout=120)
        finally:
            for image in images:
                image.close()

    def load_document(self, path):
        path = Path(path).resolve()
        relative = path.relative_to(self.data_dir)
        digest = file_hash(path)
        doc_id = document_id(relative, self.namespace)
        units = []
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                raise ValueError("PDF contains no pages")
            for index, page in enumerate(pdf.pages):
                extraction_error = None
                try:
                    text = page.extract_text() or ""
                except Exception as error:
                    extraction_error = error
                    text = ""
                    logger.warning("Extraction failed: %s page %s (%s); trying OCR",
                                   relative, index + 1, type(error).__name__)
                if self.ocr and (extraction_error or len(text.strip()) < 30):
                    try:
                        ocr_text = self._ocr_page(path, index)
                    except Exception as error:
                        raise ValueError(f"OCR failed: {relative} page {index + 1} ({type(error).__name__})") from error
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                elif extraction_error:
                    raise ValueError(f"Extraction failed: {relative} page {index + 1}") from extraction_error
                if not text.strip():
                    raise ValueError(f"No text extracted: {relative} page {index + 1}")
                units.append(DocUnit(doc_id, "pdf", (pdf.metadata or {}).get("Title") or path.stem,
                                     relative.as_posix(), index + 1, text, {"file": relative.as_posix()}))
        if file_hash(path) != digest:
            raise ValueError(f"PDF changed during extraction: {relative}")
        return Document(doc_id, digest, units)

    def iter_documents(self, files=None, limit=None):
        if not self.data_dir.is_dir():
            raise ValueError("DATA_DIR does not exist")
        paths = ([self.data_dir / name for name in files] if files else
                 sorted(p for p in self.data_dir.rglob("*") if p.suffix.lower() == ".pdf"))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise ValueError("No PDF files found")
        for path in paths:
            try:
                yield self.load_document(path)
            except Exception as error:
                self.failures.append({"file": path.name, "error": str(error)})
                logger.error("Document rejected: %s (%s)", path.name, error)


def clean_legal_text(text):
    # Preserve printable Unicode, including %, R$, comparison/math operators and sections.
    text = ftfy.fix_text(text or "", normalization="NFC")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0e-\x1f\x7f\u200b-\u200d\ufeff]", "", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


PREF_SEPARATORS = [
    r"(?im:\n(?=[ \t]*(?:CAP[ÍI]TULO|T[ÍI]TULO|SE[ÇC][ÃA]O)\b))",
    r"(?im:\n(?=[ \t]*Art(?:igo)?\.?[ \t]*\d+))",
]
BASIC_SEPARATORS = [r"\n\n", r"\n", r"\. ", r"; ", " ", ""]


def make_splitter(chunk_size, overlap):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len,
        separators=PREF_SEPARATORS + BASIC_SEPARATORS,
        is_separator_regex=True, keep_separator=True,
    )


def iter_chunks(document, settings):
    splitter = make_splitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    for unit in document.units:
        cleaned = clean_legal_text(unit.text)
        if not cleaned:
            raise ValueError(f"Page {unit.page} contains no indexable text")
        for chunk in splitter.split_text(cleaned):
            yield {"content": chunk, "source": unit.source, "title": unit.title,
                   "uri": unit.uri, "page": unit.page}


def document_version(document, settings):
    description = {
        "source_hash": document.source_hash, "pipeline": PIPELINE_VERSION,
        "chunk_size": settings.CHUNK_SIZE, "overlap": settings.CHUNK_OVERLAP,
        # OCR output and titles can vary without changes to the PDF bytes.
        "pages": [(unit.page, unit.title, clean_legal_text(unit.text)) for unit in document.units],
    }
    return hashlib.sha256(json.dumps(description, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def embedding_batches(document, encoder, settings):
    payloads = []
    batch_size = min(settings.QDRANT_BATCH_SIZE, settings.EMBEDDING_BATCH_SIZE)
    for payload in iter_chunks(document, settings):
        payloads.append(payload)
        if len(payloads) == batch_size:
            yield encoder.generate_batch([p["content"] for p in payloads]), payloads
            payloads = []
    if payloads:
        yield encoder.generate_batch([p["content"] for p in payloads]), payloads


def ingest_documents(loader, manager, encoder, settings, files=None, limit=None):
    dimension = encoder.model.get_sentence_embedding_dimension()
    manager.ensure_collection_exists(vector_size=dimension)
    report = {"documents": 0, "chunks": 0, "failed": []}
    for document in loader.iter_documents(files=files, limit=limit):
        try:
            count = manager.publish_document(
                document.doc_id, document_version(document, settings),
                embedding_batches(document, encoder, settings), dimension, document.source_hash,
            )
            report["documents"] += 1
            report["chunks"] += count
        except Exception as error:
            logger.error("Indexing failed for %s (%s)", document.doc_id, type(error).__name__)
            report["failed"].append({"doc_id": document.doc_id, "error": type(error).__name__})
    report["failed"].extend(loader.failures)
    if report["documents"] == 0 or report["failed"]:
        raise RuntimeError("Ingestion incomplete: " + json.dumps(report, ensure_ascii=False))
    return report


def run_ingestion(collection, *, files=None, limit=None, ocr=True):
    settings = get_settings()
    if not collection or collection == "juribot_chunks":
        raise ValueError("Choose an explicit new collection; the legacy juribot_chunks is protected")
    # Prevent overlapping writers on this host. Use one ingestion host per collection.
    key = hashlib.sha256(f"{settings.QDRANT_URL or settings.QDRANT_HOST}:{collection}".encode()).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"juribot-ingest-{key}.lock"
    with portalocker.Lock(str(lock_path), timeout=0):
        encoder = EmbeddingGenerator(settings)
        manager = QdrantManager(collection, settings=settings)
        try:
            loader = PDFLoader(settings.DATA_DIR, ocr, settings.POPPLER_PATH, settings.DOCUMENT_NAMESPACE)
            report = ingest_documents(loader, manager, encoder, settings, files, limit)
            print(json.dumps({"collection": collection, **report}, ensure_ascii=False))
            return report
        finally:
            manager.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, help="New or previously validated v2 collection")
    parser.add_argument("--file", action="append", help="PDF path relative to DATA_DIR; repeatable")
    parser.add_argument("--limit", type=int, help="Maximum PDFs to process")
    parser.add_argument("--no-ocr", action="store_true", help="Only use embedded PDF text")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_ingestion(args.collection, files=args.file, limit=args.limit, ocr=not args.no_ocr)
    except Exception as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
