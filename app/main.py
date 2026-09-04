from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Enterprise GenAI Assistant"
)

app.include_router(router)