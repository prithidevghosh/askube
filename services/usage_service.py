from config.db import usage_collection

# ── Pricing constants (update here if rates change) ──────────────
# Claude Haiku 4.5
ANTHROPIC_INPUT_PRICE_PER_TOKEN  = 0.80  / 1_000_000   # $0.80  per 1M input tokens
ANTHROPIC_OUTPUT_PRICE_PER_TOKEN = 4.00  / 1_000_000   # $4.00  per 1M output tokens

# OpenAI text-embedding-3-small
OPENAI_EMBEDDING_PRICE_PER_TOKEN = 0.020 / 1_000_000   # $0.020 per 1M tokens


async def create_usage_entry(cs_id: str, v_id: str, video_length_seconds: float) -> None:
    """Insert a blank usage record when a session is created."""
    await usage_collection.insert_one({
        "cs_id": cs_id,
        "v_id": v_id,
        "video_length_seconds": round(video_length_seconds, 2),
        "anthropic": {
            "input_tokens":  0,
            "output_tokens": 0,
            "cost_usd":      0.0,
        },
        "openai": {
            "embedding_tokens": 0,
            "cost_usd":         0.0,
        },
        "total_cost_usd": 0.0,
    })


async def add_anthropic_usage(cs_id: str, input_tokens: int, output_tokens: int) -> None:
    """Increment Anthropic token counts and recalculate cost for a session."""
    input_cost  = input_tokens  * ANTHROPIC_INPUT_PRICE_PER_TOKEN
    output_cost = output_tokens * ANTHROPIC_OUTPUT_PRICE_PER_TOKEN
    call_cost   = input_cost + output_cost

    await usage_collection.update_one(
        {"cs_id": cs_id},
        {"$inc": {
            "anthropic.input_tokens":  input_tokens,
            "anthropic.output_tokens": output_tokens,
            "anthropic.cost_usd":      call_cost,
            "total_cost_usd":          call_cost,
        }},
    )


async def add_openai_usage(cs_id: str, tokens: int) -> None:
    """Increment OpenAI embedding token count and recalculate cost for a session."""
    call_cost = tokens * OPENAI_EMBEDDING_PRICE_PER_TOKEN

    await usage_collection.update_one(
        {"cs_id": cs_id},
        {"$inc": {
            "openai.embedding_tokens": tokens,
            "openai.cost_usd":         call_cost,
            "total_cost_usd":          call_cost,
        }},
    )
