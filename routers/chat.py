import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat_service import get_chat_session, stream_chat_response

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    cs_id: str
    v_id: str
    user_q: str


@router.post("/stream")
async def chat_stream(body: ChatRequest):
    session = await get_chat_session(body.cs_id, body.v_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    async def _json_wrap():
        async for chunk in stream_chat_response(body.cs_id, body.v_id, body.user_q):
            yield json.dumps({"assistant_reply": chunk}) + "\n"

    return StreamingResponse(_json_wrap(), media_type="application/x-ndjson")
