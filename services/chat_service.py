import anthropic
from openai import AsyncOpenAI
from typing import AsyncGenerator

from config.db import chats_collection
from config.chroma import chroma_client
from services.usage_service import add_anthropic_usage, add_openai_usage

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
TOP_K_CHUNKS = 3

# Budget for Q&A history sent to Claude (~15k tokens).
# MongoDB stores the full history; only the trimmed window goes to the API.
MAX_HISTORY_CHARS = 60_000

claude_client = anthropic.AsyncAnthropic()
openai_client = AsyncOpenAI()

SYSTEM_PERSONA = """You are Askube, an intelligent video assistant.

Answer the user's question based ONLY on the transcript excerpts provided below.
When referencing specific moments, include the timestamp shown.
If the answer is not in the excerpts, say: "I couldn't find that in the video. Try rephrasing your question."

Strict scope rule:
- If a question is NOT related to the video content, politely decline and say:
  "That question seems to be outside the scope of this video. Feel free to ask me anything about the video!"

Tone: Friendly, clear, and concise.

--- RELEVANT TRANSCRIPT EXCERPTS ---
{context}
--- END OF EXCERPTS ---
"""

REWRITE_PROMPT = """Given the conversation history below, rewrite the latest user question as a short, standalone search phrase that captures the actual topic. Use words likely to appear in a spoken video transcript. Output ONLY the rewritten query, nothing else.

Conversation history:
{history}

Latest question: {question}"""


async def _rewrite_query(user_q: str, qa_history: list[dict], cs_id: str) -> str:
    """Rewrite follow-up questions into standalone search phrases using last 6 messages."""
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in qa_history[-6:]
    )
    response = await claude_client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(
            history=history_text,
            question=user_q,
        )}],
    )
    await add_anthropic_usage(cs_id, response.usage.input_tokens, response.usage.output_tokens)
    rewritten = response.content[0].text.strip()
    print(f"Query rewrite: {user_q!r} → {rewritten!r}")
    return rewritten


async def _retrieve_chunks(v_id: str, query: str, cs_id: str) -> str:
    """Embed query and retrieve top-K semantically similar chunks from ChromaDB."""
    response = await openai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small",
    )
    await add_openai_usage(cs_id, response.usage.total_tokens)
    query_vector = response.data[0].embedding

    collection = chroma_client.get_collection(f"video_{v_id}")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K_CHUNKS,
        include=["documents", "metadatas"],
    )

    lines = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        lines.append(f"[{meta['timestamp']}] {doc}")
    return "\n\n".join(lines)


def _trim_qa_history(qa_history: list[dict]) -> list[dict]:
    """Drop oldest user/assistant pairs until within MAX_HISTORY_CHARS. Never drops last entry."""
    while len(qa_history) > 1:
        total = sum(len(str(m["content"])) for m in qa_history)
        if total <= MAX_HISTORY_CHARS:
            break
        qa_history = qa_history[2:]
    return qa_history


async def get_chat_session(cs_id: str, v_id: str) -> dict | None:
    return await chats_collection.find_one({"cs_id": cs_id, "v_id": v_id})


async def stream_chat_response(cs_id: str, v_id: str, user_q: str) -> AsyncGenerator[str, None]:
    """
    RAG pipeline:
      1. Load conversation history from MongoDB.
      2. Rewrite query if history exists, else use raw query.
      3. Embed query → retrieve top-K chunks from ChromaDB.
      4. Stream Claude response with retrieved chunks as context.
      5. Persist user message and assistant reply to MongoDB.
    """
    chat_session = await chats_collection.find_one({"cs_id": cs_id, "v_id": v_id})
    qa_history = chat_session["past_conversation"]

    # Step 1: rewrite if follow-up, use raw query for first message
    search_query = await _rewrite_query(user_q, qa_history, cs_id) if qa_history else user_q

    # Step 2: retrieve relevant chunks
    context = await _retrieve_chunks(v_id, search_query, cs_id)
    system_prompt = SYSTEM_PERSONA.format(context=context)

    # Step 3: build messages for Claude (trim to token budget)
    messages = _trim_qa_history(list(qa_history))
    messages.append({"role": "user", "content": user_q})

    # Persist user message
    await chats_collection.update_one(
        {"cs_id": cs_id, "v_id": v_id},
        {"$push": {"past_conversation": {"role": "user", "content": user_q}}}
    )

    # Step 4: stream Claude response
    full_response: list[str] = []
    async with claude_client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            full_response.append(text)
            yield text
        final = await stream.get_final_message()
        await add_anthropic_usage(cs_id, final.usage.input_tokens, final.usage.output_tokens)

    # Persist assistant reply
    await chats_collection.update_one(
        {"cs_id": cs_id, "v_id": v_id},
        {"$push": {"past_conversation": {"role": "assistant", "content": "".join(full_response)}}}
    )
