# backend/app_main.py
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


def create_app() -> FastAPI:
    """Cria a aplicação FastAPI com CORS e /health."""
    app = FastAPI(
        title="Juribot API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Somente interfaces locais durante a estabilização, sem cookies de terceiros.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health", tags=["system"])
    def health():
        return {"status": "ok"}

    return app


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = create_app()

# -----------------------------------------------------------------------------
# Rotas externas opcionais (/chat, /classify)
# -----------------------------------------------------------------------------
def _include_chat_router(app):
    # tenta "api.routers_chat" quando o WORKDIR é /app/backend
    try:
        from api.routers_chat import router as chat_router
        app.include_router(chat_router)
        print("[BOOT] /chat incluído via api.routers_chat")
        return
    except Exception as e:
        print(f"[WARN] /chat não incluído por api.routers_chat: {e}")

    # fallback: quando o Python path espera backend.api.routers_chat
    try:
        from backend.api.routers_chat import router as chat_router
        app.include_router(chat_router)
        print("[BOOT] /chat incluído via backend.api.routers_chat")
        return
    except Exception as e:
        print(f"[WARN] /chat não incluído por backend.api.routers_chat: {e}")

_include_chat_router(app)

def _include_classify_router(app):
    try:
        from api.routes_classify import router as classify_router
        app.include_router(classify_router)
        print("[BOOT] /classify incluído via api.routes_classify")
        return
    except Exception as e:
        print(f"[WARN] /classify não incluído por api.routes_classify: {e}")

    try:
        from backend.api.routes_classify import router as classify_router
        app.include_router(classify_router)
        print("[BOOT] /classify incluído via backend.api.routes_classify")
        return
    except Exception as e:
        print(f"[WARN] /classify não incluído por backend.api.routes_classify: {e}")

_include_classify_router(app)


# -----------------------------------------------------------------------------
# Boot de dependências (carregadas 1x em app.state)
# -----------------------------------------------------------------------------
# Descoberta de device para embeddings
_device = os.getenv("EMBEDDING_DEVICE", "auto").lower()
if _device == "auto":
    try:
        import torch  # type: ignore
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _cuda_name = torch.cuda.get_device_name(0) if _device == "cuda" else "N/A"
    except Exception:
        _device, _cuda_name = "cpu", "N/A"
else:
    _cuda_name = "N/A"

# SentenceTransformer (encoder) — carrega uma única vez
if not hasattr(app.state, "model"):
    _model_id = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    app.state.model = SentenceTransformer(_model_id, device=_device)
    print(f"[BOOT] Embeddings: {_model_id} | device={_device} | CUDA={_cuda_name}")

# Qdrant client — carrega uma única vez
if not hasattr(app.state, "qdrant"):
    app.state.qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
        prefer_grpc=os.getenv("QDRANT_USE_GRPC", "true").lower() == "true",
        timeout=float(os.getenv("QDRANT_TIMEOUT", "180")),
    )
    print(
        "[BOOT] Qdrant:",
        os.getenv("QDRANT_HOST", "qdrant"),
        os.getenv("QDRANT_PORT", "6333"),
        os.getenv("QDRANT_GRPC_PORT", "6334"),
        "| prefer_grpc=" + str(os.getenv("QDRANT_USE_GRPC", "true")),
    )

# -----------------------------------------------------------------------------
# Rotas externas opcionais (/chat, /classify)
# -----------------------------------------------------------------------------
try:
    from api.routers_chat import router as chat_router  # seu existente
    app.include_router(chat_router)
except Exception as e:
    print(f"[WARN] /chat não incluído: {e}")

try:
    from api.routes_classify import router as classify_router  # se existir
    app.include_router(classify_router)
except Exception as e:
    print(f"[WARN] /classify não incluído: {e}")

# -----------------------------------------------------------------------------
# UI estática e redirect raiz
# -----------------------------------------------------------------------------
if os.path.isdir("ui"):
    app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/ui/") if os.path.isdir("ui") else RedirectResponse(url="/docs")

# -----------------------------------------------------------------------------
# /search (RAG simples)
# -----------------------------------------------------------------------------
class SearchReq(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search", tags=["search"])
def search(req: SearchReq):
    vec = app.state.model.encode(req.query).tolist()
    hits = app.state.qdrant.search(
        collection_name=os.getenv("QDRANT_COLLECTION", "juribot_chunks"),
        query_vector=vec,
        limit=req.top_k,
        with_payload=True,
    )
    results = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "score": float(h.score),
                "title": payload.get("title"),
                "page": payload.get("page"),
                "uri": payload.get("uri"),
                "snippet": (payload.get("content") or "")[:400],
            }
        )
    return results
