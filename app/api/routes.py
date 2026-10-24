from fastapi import APIRouter
from pydantic import BaseModel

from app.memory.chat_memory import clear_history
from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


@router.get("/")
def home():
    return {"status": "running"}


@router.post("/chat")
def chat(request: ChatRequest):
    return chat_service.chat(request.question, request.session_id)


@router.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    clear_history(session_id)
    return {"status": "cleared"}