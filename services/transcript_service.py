import re
import random
import string
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from config.db import transcript_collection, chats_collection
from services.embedding_service import embed_and_store
from services.usage_service import create_usage_entry


def parse_youtube_url(url):
    video_id_match = re.search(
        r'(?:youtube\.com\/.*?[?&]v=|youtu\.be\/)([^"&?\/\s]{11})', url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        raise ValueError('Invalid YouTube URL')


def _generate_cs_id() -> str:
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=25))


async def fetch_transcript(youtube_url: str) -> tuple[str, str]:
    """
    Fetches (or retrieves cached) transcript, creates a new chat session,
    and returns (v_id, cs_id).
    """
    video_id = parse_youtube_url(youtube_url)
    print(f"Extracted video ID: {video_id}")

    existing = await transcript_collection.find_one({"video_id": video_id}, {"_id": 0, "transcript": 1})
    if existing:
        print(f"Cache hit for video ID: {video_id}")
        transcript = existing["transcript"]
    else:
        api = YouTubeTranscriptApi()
        transcript_obj = api.fetch(video_id)
        transcript = transcript_obj.to_raw_data()
        for snippet in transcript:
            snippet["text"] = snippet["text"].replace('"', '')

        await transcript_collection.insert_one({
            "video_id": video_id,
            "transcript": transcript,
        })

    embed_and_store(video_id, transcript)

    # Video length = last entry's start + duration
    last = transcript[-1]
    video_length = last["start"] + last.get("duration", 0)

    cs_id = _generate_cs_id()
    await chats_collection.insert_one({
        "cs_id": cs_id,
        "v_id": video_id,
        "past_conversation": [],
    })
    await create_usage_entry(cs_id, video_id, video_length)

    return video_id, cs_id
