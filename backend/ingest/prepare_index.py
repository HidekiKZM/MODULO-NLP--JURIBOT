# Arquivo: backend/ingest/prepare_index.py

import hashlib
import ftfy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
import pytesseract

# Importa componentes do projeto
from backend.core.config import get_settings
from backend.rag.embeddings import EmbeddingGenerator
from backend.rag.vector_qdrant import QdrantManager


# ==========================
# Tipos e utilitários
# ==========================

@dataclass
class DocUnit:
    doc_id: str
    source: str
    title: str
    uri: str
    page: Optional[int]
    text: str
    extra: Dict

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# ==========================
# Loaders
# ==========================

class PDFLoader:
    """
    Carrega PDFs de data_dir, extraindo texto por página.
    Se a página tiver pouco texto, tenta OCR (se ocr=True).
    """
    def __init__(self, data_dir: Path, ocr: bool = True, poppler_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.ocr = ocr
        # Opcional: Definir caminho do poppler para Windows
        self.poppler_path = poppler_path or get_settings().POPPLER_PATH

    def _extract_page_text(self, pdf: pdfplumber.PDF, page_idx: int) -> str:
        try:
            page = pdf.pages[page_idx]
            text = page.extract_text() or ""
            return text
        except Exception:
            return ""

    def _ocr_page(self, pdf_path: Path, page_num: int) -> str:
        """Realiza OCR da página (page_num é 0-based)."""
        try:
            images = convert_from_path(
                str(pdf_path),
                first_page=page_num + 1,
                last_page=page_num + 1,
                poppler_path=self.poppler_path
            )
            if not images:
                return ""
            img = images[0]
            return pytesseract.image_to_string(img, lang="por+eng")
        except Exception:
            return ""

    def load(self) -> List[DocUnit]:
        docs: List[DocUnit] = []
        if not self.data_dir.exists():
            print(f"⚠️  Diretório de dados não existe: {self.data_dir}")
            return docs

        pdf_files = sorted(self.data_dir.rglob("*.pdf"))
        for pdf_path in pdf_files:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    num_pages = len(pdf.pages)
                    for i in range(num_pages):
                        text = self._extract_page_text(pdf, i)
                        # Se texto parece curto e OCR habilitado, tentar OCR
                        if self.ocr and len((text or "").strip()) < 30:
                            ocr_text = self._ocr_page(pdf_path, i)
                            if len(ocr_text.strip()) > len(text.strip()):
                                text = ocr_text

                        # Pula páginas totalmente vazias
                        if not (text and text.strip()):
                            continue

                        title = pdf.metadata.get("Title") or pdf_path.stem
                        doc_id = sha1(f"{pdf_path.as_posix()}::{i}")
                        docs.append(DocUnit(
                            doc_id=doc_id,
                            source="pdf",
                            title=title,
                            uri=pdf_path.as_posix(),
                            page=i + 1,
                            text=text,
                            extra={"file": pdf_path.name}
                        ))
            except Exception as e:
                print(f"⚠️  Falha ao ler PDF '{pdf_path}': {e}")
        return docs


class WebLoader:
    """
    Carrega páginas web a partir de uma lista de URLs, usando trafilatura para extrair o texto limpo.
    """
    def __init__(self, urls: List[str]):
        self.urls = urls or []

    def _fetch(self, url: str) -> Optional[str]:
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            if not downloaded:
                return None
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return text
        except Exception:
            return None

    def load(self) -> List[DocUnit]:
        docs: List[DocUnit] = []
        for url in self.urls:
            text = self._fetch(url)
            if not (text and text.strip()):
                continue

            # Tentar título básico via requests + BS4 (opcional)
            title = ""
            try:
                resp = requests.get(url, timeout=10)
                if resp.ok:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    ttag = soup.find("title")
                    title = (ttag.text or "").strip() if ttag else ""
            except Exception:
                pass

            title = title or url
            doc_id = sha1(url)
            docs.append(DocUnit(
                doc_id=doc_id,
                source="web",
                title=title,
                uri=url,
                page=None,
                text=text,
                extra={}
            ))
        return docs


# ==========================
# Limpeza e chunking
# ==========================

LEGAL_KEEP = "§ºª№°“”–—«»/\\"
LEGAL_RE = re.compile(rf"[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\n\r{re.escape(LEGAL_KEEP)}]", re.UNICODE)

def clean_legal_text(text: str) -> str:
    """
    Corrige mojibake/acentuação e normaliza o texto para o pipeline jurídico.
    """
    if not text:
        return ""

    # Corrige mojibake e normaliza acentuação (NFC)
    text = ftfy.fix_text(text, normalization="NFC")

    # Normaliza quebras de linha e espaços
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)

    # Remove caracteres invisíveis (zero-width, BOM)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # Aplica o filtro de caracteres permitido (mantém acentos por causa do re.UNICODE em LEGAL_RE)
    text = LEGAL_RE.sub("", text)

    # Compacta quebras de linha extras e espaços antes de \n
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



PREF_SEPARATORS = [
    r"\n(?=Art\.?\s*\d+)", r"\n(?=CAP[ÍI]TULO)", r"\n(?=SEÇ[ÃA]O)", r"\n(?=T[ÍI]TULO)",
]
BASIC_SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]

def make_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=PREF_SEPARATORS + BASIC_SEPARATORS + [""],
    )

def split_docs(docs: List[DocUnit], chunk_size: int, overlap: int) -> Tuple[List[str], List[Dict]]:
    splitter = make_splitter(chunk_size, overlap)
    out_texts: List[str] = []
    out_payloads: List[Dict] = []

    for d in docs:
        cleaned = clean_legal_text(d.text)
        if not cleaned:
            continue
        chunks = splitter.split_text(cleaned)
        for i, c in enumerate(chunks):
            out_texts.append(c)
            out_payloads.append({
                "content": c,  # payload precisa conter o próprio conteúdo
                "source": d.source,
                "title": d.title,
                "uri": d.uri,
                "page": d.page,
                "chunk_no": i,
                "doc_id": d.doc_id,
            })
    return out_texts, out_payloads


# ==========================
# Pipeline principal
# ==========================

def run_ingestion():
    """
    Orquestra o processo completo de ingestão de documentos.
    """
    print("🚀 Iniciando o processo de ingestão de documentos...")

    # A pasta de dados relativa à raiz do projeto (monte no Docker se for o caso)
    settings = get_settings()
    data_path = settings.DATA_DIR

    # 1) Carrega documentos
    print(f"📂 Carregando documentos de '{data_path.resolve()}'...")
    pdf_loader = PDFLoader(data_dir=data_path, ocr=True)
    pdf_docs = pdf_loader.load()
    print(f"📄 Encontrados {len(pdf_docs)} páginas de PDF.")

    # Se quiser ativar web:
    # urls = ["https://www.estrategiaconcursos.com.br/blog/cdc-codigo-defesa-consumidor/"]
    # web_docs = WebLoader(urls).load()
    # print(f"🌐 Encontrados {len(web_docs)} documentos da web.")
    # all_docs = pdf_docs + web_docs

    all_docs = pdf_docs

    if not all_docs:
        print("⚠️ Nenhum documento encontrado. Encerrando a ingestão.")
        return

    # 2) Limpeza e chunking
    print("\n🔄 Limpando e dividindo documentos em chunks...")
    texts, payloads = split_docs(
        all_docs,
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP,
    )
    print(f"✅ Documentos divididos em {len(texts)} chunks.")

    # 3) Embeddings
    print("\n🧠 Gerando embeddings (isso pode levar um tempo)...")
    encoder = EmbeddingGenerator()
    embeddings = encoder.generate_batch(texts)
    if not embeddings:
        raise RuntimeError("Falha ao gerar embeddings (lista vazia).")
    dim = len(embeddings[0]) if embeddings else 0
    print(f"✅ Embeddings gerados com sucesso. Shape: ({len(embeddings)}, {dim})")

    # 4) Upsert no Qdrant
    print("\n💾 Inserindo dados no banco vetorial Qdrant...")
    qdrant_manager = QdrantManager()
    qdrant_manager.ensure_collection_exists(vector_size=dim)
    qdrant_manager.upsert_points(vectors=embeddings, payloads=payloads)
    print("✅ Dados inseridos no Qdrant com sucesso!")

    print("\n🏁 Processo de ingestão concluído!")


if __name__ == "__main__":
    # A partir da raiz: python -m backend.ingest.prepare_index
    run_ingestion()
