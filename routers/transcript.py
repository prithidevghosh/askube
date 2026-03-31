from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from services.transcript_service import fetch_transcript
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, IpBlocked

router = APIRouter(prefix="/transcript", tags=["transcript"])


class TranscriptUploadRequest(BaseModel):
    youtube_url: HttpUrl


@router.post("/upload")
async def upload_transcript(body: TranscriptUploadRequest):
    try:
        v_id, cs_id = await fetch_transcript(str(body.youtube_url))
        return {"v_id": v_id, "cs_id": cs_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IpBlocked:
        raise HTTPException(status_code=503, detail="YouTube is blocking requests from this IP.")
    except TranscriptsDisabled:
        raise HTTPException(status_code=422, detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=404, detail="No transcript found for this video.")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable.")
