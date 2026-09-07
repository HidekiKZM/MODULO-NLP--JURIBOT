"""Chat uses resources initialized once by the application lifespan."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.core.runtime import SearchUnavailable

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatReq(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=6, ge=1, le=20)
    label_hint: Optional[str] = None


@router.post("")
@router.post("/", include_in_schema=False)
def chat(req: ChatReq, request: Request):
    runtime = request.app.state.runtime
    if runtime is None:
        raise HTTPException(status_code=503, detail="Inicialização pendente")
    settings = request.app.state.settings
    decision = request.app.state.rag_router.decide(req.question, classified_label=req.label_hint)
    mode = decision.get("mode", "rag")
    sources = []
    if mode == "rag":
        try:
            sources = runtime.search(settings, req.question, req.top_k)
        except SearchUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    if not settings.gemini_enabled:
        return {
            "answer": (
                "LLM não configurada. Aqui estão as fontes mais relevantes."
                if mode == "rag" else
                "LLM não configurada. Configure GEMINI_API_KEY e GEMINI_MODEL para respostas diretas."
            ),
            "sources": sources,
            "routing": decision,
        }
    if runtime.gemini is None:
        raise HTTPException(status_code=503, detail="Geração indisponível; consulte /ready")

    if mode == "rag":
        context = "\n\n".join(f"[{i+1}] {source.get('snippet', '')}" for i, source in enumerate(sources))
        prompt = (
            "Você é um assistente jurídico. Responda de forma objetiva e cite as fontes "
            "entre colchetes como [1], [2] quando forem utilizadas.\n\n"
            f"FONTES:\n{context}\n\nPERGUNTA: {req.question}\n\nRESPOSTA:"
        )
    else:
        prompt = (
            "Você é um assistente jurídico. Responda de forma objetiva, baseada em conhecimento geral. "
            "Se necessário, sugira buscar fontes.\n\n"
            f"PERGUNTA: {req.question}\n\nRESPOSTA:"
        )
    try:
        response = runtime.gemini.generate_content(prompt)
        answer = (getattr(response, "text", "") or "").strip() or "Não consegui gerar resposta. Tente reformular a pergunta."
    except Exception as error:
        logger.warning("Falha na geração (%s)", type(error).__name__)
        raise HTTPException(status_code=503, detail="Falha ao gerar resposta com Gemini") from error
    return {"answer": answer, "sources": sources, "routing": decision}
