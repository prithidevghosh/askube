import anthropic
from typing import AsyncGenerator

from config.db import chats_collection, transcript_collection

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

# ~4 chars per token; leave ample room for conversation history and response
MAX_TRANSCRIPT_CHARS = 120_000

# Budget for Q&A history sent to Claude (~15k tokens).
# MongoDB stores the full history; only the trimmed window goes to the API.
MAX_HISTORY_CHARS = 60_000

client = anthropic.AsyncAnthropic()

SYSTEM_PERSONA = """You are Askube, an intelligent video assistant with deep expertise in the content of the video transcript provided to you.

Your role:
- Develop thorough knowledge and understanding of the entire video content.
- Answer any question a user has about the video — summaries, specific moments, themes, speaker intent, explanations, comparisons, etc.
- For timestamp-based questions (e.g. "what happens between 2:24 and 2:38?"), locate the relevant segments from the transcript using the start times and answer precisely.
- If the user asks for a summary, provide a concise but comprehensive overview of the video.

Strict scope rule:
- If a question is NOT related to the video content, politely decline and say:
  "That question seems to be outside the scope of this video. I can only answer questions related to the video content. Feel free to ask me anything about the video!"

Tone: Friendly, clear, and informative. Keep answers concise but complete.

--- VIDEO TRANSCRIPT ---
{transcript_text}
--- END OF TRANSCRIPT ---
"""


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for snippet in transcript:
        start = snippet.get("start", 0)
        minutes = int(start // 60)
        seconds = int(start % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        text = snippet.get("text", "").strip()
        if text:
            lines.append(f"[{timestamp}] {text}")
    return "\n".join(lines)


def _chunk_transcript(transcript: list[dict]) -> str:
    full_text = _format_transcript(transcript)
    if len(full_text) <= MAX_TRANSCRIPT_CHARS:
        return full_text

    truncated = full_text[:MAX_TRANSCRIPT_CHARS]
    last_newline = truncated.rfind("\n")
    if last_newline != -1:
        truncated = truncated[:last_newline]
    truncated += "\n[Transcript truncated due to length]"
    return truncated


def _trim_qa_history(qa_history: list[dict]) -> list[dict]:
    """
    Drop oldest user/assistant pairs until total character count is within
    MAX_HISTORY_CHARS. The last entry (the new user question) is never dropped.
    """
    while len(qa_history) > 1:
        total = sum(len(str(m["content"])) for m in qa_history)
        if total <= MAX_HISTORY_CHARS:
            break
        # Drop the oldest pair (user at [0], assistant at [1])
        qa_history = qa_history[2:]
    return qa_history


async def get_chat_session(cs_id: str, v_id: str) -> dict | None:
    return await chats_collection.find_one({"cs_id": cs_id, "v_id": v_id})


async def stream_chat_response(cs_id: str, v_id: str, user_q: str) -> AsyncGenerator[str, None]:
    """
    Streams the assistant reply for a given chat session.
    Persists both the user message and the final assistant reply to chats_data.
    Caller must verify the session exists before calling this.
    """
    chat_session = await chats_collection.find_one({"cs_id": cs_id, "v_id": v_id})

    transcript_doc = await transcript_collection.find_one(
        {"video_id": v_id}, {"_id": 0, "transcript": 1}
    )
    transcript_text = _chunk_transcript(transcript_doc["transcript"])
    system_prompt = SYSTEM_PERSONA.format(transcript_text=transcript_text)

    # past_conversation[0] = initial user upload (transcript data)
    # past_conversation[1] = assistant acknowledgement
    # past_conversation[2:] = actual Q&A history
    qa_history = [
        {"role": entry["role"], "content": entry["content"]}
        for entry in chat_session["past_conversation"][2:]
    ]
    qa_history.append({"role": "user", "content": user_q})
    qa_history = _trim_qa_history(qa_history)

    # Persist user message immediately
    await chats_collection.update_one(
        {"cs_id": cs_id, "v_id": v_id},
        {"$push": {"past_conversation": {"role": "user", "content": user_q}}}
    )

    # Stream response and collect for persistence
    full_response: list[str] = []
    async with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=qa_history,
    ) as stream:
        async for text in stream.text_stream:
            full_response.append(text)
            yield text

    # Persist assistant reply
    await chats_collection.update_one(
        {"cs_id": cs_id, "v_id": v_id},
        {"$push": {"past_conversation": {"role": "assistant", "content": "".join(full_response)}}}
    )
