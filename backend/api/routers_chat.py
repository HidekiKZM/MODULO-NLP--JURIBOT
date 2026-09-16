"""HTTP validation and error mapping; ChatService owns the chat flow."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from backend.services.chat_errors import ChatFailure

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatReq(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=6, ge=1, le=20)
    label_hint: Optional[str] = Field(default=None, max_length=128)


@router.post("")
@router.post("/", include_in_schema=False)
async def chat(req: ChatReq, request: Request):
    try:
        service = request.app.state.chat_service
        if service is None:
            raise ChatFailure("chat_initializing")
        return await service.reply(req.question, req.top_k, req.label_hint)
    except ChatFailure as error:
        raise HTTPException(status_code=error.status, detail=error.detail()) from error
