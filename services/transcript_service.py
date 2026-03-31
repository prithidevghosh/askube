import re
import random
import string
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from config.db import transcript_collection, chats_collection


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

    cs_id = _generate_cs_id()
    initial_conversation = [
        {"role": "user", "content": transcript},
        {"role": "assistant", "content": "Thanks for the url, I have analyzed it, you can ask me any questions from this video"},
    ]
    await chats_collection.insert_one({
        "cs_id": cs_id,
        "v_id": video_id,
        "past_conversation": initial_conversation,
    })

    return video_id, cs_id
