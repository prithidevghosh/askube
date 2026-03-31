from fastapi import FastAPI

from routers import transcript
from routers import chat

app = FastAPI(title="Askube API", version="1.0.0")

app.include_router(transcript.router)
app.include_router(chat.router)
