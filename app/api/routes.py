import json

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.memory.chat_memory import clear_history
from app.rag.vectorstore import list_documents, save_and_index
from app.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


@router.get("/")
def home():
    return {"status": "running"}


# ---------------------------------
# Chat
# ---------------------------------

@router.post("/chat")
def chat(request: ChatRequest):
    return chat_service.chat(request.question, request.session_id)


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):

    def events():
        for event in chat_service.stream_chat(
            request.question,
            request.session_id,
        ):
            # One JSON object per line (NDJSON)
            yield json.dumps(event) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    clear_history(session_id)
    return {"status": "cleared"}


# ---------------------------------
# Documents
# ---------------------------------

@router.get("/documents")
def documents():
    return {"documents": list_documents()}


@router.post("/documents")
def upload_document(file: UploadFile = File(...)):
    data = file.file.read()

    try:
        return save_and_index(chat_service.vectorstore, file.filename, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))