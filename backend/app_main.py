"""Run from the repository root: python -m uvicorn backend.app_main:app."""
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Required routes use explicit imports: an import failure must stop startup.
from backend.api.routers_chat import router as chat_router
from backend.api.routes_classify import router as classify_router
from backend.core.config import Settings, UI_DIR, get_settings
from backend.core.runtime import Runtime, SearchUnavailable, load_runtime
from backend.services.router import RAGRouter


class SearchReq(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)


def create_app(
    settings: Settings | None = None,
    resource_loader: Callable[[Settings], Runtime] = load_runtime,
) -> FastAPI:
    config = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await run_in_threadpool(resource_loader, config)
        app.state.runtime = runtime
        app.state.rag_router = RAGRouter(runtime.model, runtime.qdrant)
        try:
            yield
        finally:
            await run_in_threadpool(runtime.close)
            app.state.runtime = None
            app.state.rag_router = None

    app = FastAPI(title="Juribot API", version="0.1.0", lifespan=lifespan)
    app.state.settings = config
    app.state.runtime = None

    # Somente interfaces locais durante a estabilização, sem cookies de terceiros.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8000", "http://127.0.0.1:8000",
            "http://localhost:3000", "http://127.0.0.1:3000",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(chat_router)
    app.include_router(classify_router)
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse(url="/ui/")

    @app.get("/health", tags=["system"])
    def health():
        """Liveness: the process responds, regardless of external dependencies."""
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    def ready(request: Request):
        """Readiness: resources loaded, Qdrant reachable, compatible nonempty index."""
        runtime = request.app.state.runtime
        if runtime is None:
            return JSONResponse(status_code=503, content={"status": "not_ready", "checks": {"startup": "pending"}})
        is_ready, checks = runtime.readiness(config)
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "not_ready", "checks": checks},
        )

    @app.post("/search", tags=["search"])
    def search(req: SearchReq, request: Request):
        runtime = request.app.state.runtime
        if runtime is None:
            raise HTTPException(status_code=503, detail="Inicialização pendente")
        try:
            return runtime.search(config, req.query, req.top_k)
        except SearchUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


app = create_app()
