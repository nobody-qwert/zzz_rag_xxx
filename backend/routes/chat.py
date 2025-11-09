from __future__ import annotations

from fastapi import APIRouter

try:
    from ..schemas import AskRequest, AskResponse
    from ..services.rag import ask_question
except ImportError:  # pragma: no cover - running as top-level modules
    from schemas import AskRequest, AskResponse  # type: ignore
    from services.rag import ask_question  # type: ignore

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    return await ask_question(req)
