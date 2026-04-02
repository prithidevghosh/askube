from openai import OpenAI

from config.chroma import chroma_client

openai_client = OpenAI()


def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _clean_text(text: str) -> str:
    """Strip speaker markers and normalize whitespace."""
    import re
    text = re.sub(r'^>>\s*', '', text.strip())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_transcript(transcript: list, chunk_duration: int = 30) -> list[dict]:
    """
    Groups transcript entries into ~30s windows with a 2-entry overlap
    between consecutive chunks. Cleans speaker markers before joining.
    """
    chunks = []
    current_texts = []
    current_start = transcript[0]["start"]
    current_end = 0

    for i, entry in enumerate(transcript):
        cleaned = _clean_text(entry["text"])
        if not cleaned:
            continue
        current_texts.append(cleaned)
        current_end = entry["start"] + entry["duration"]

        if current_end - current_start >= chunk_duration:
            chunks.append({
                "text": " ".join(current_texts),
                "start": current_start,
                "end": current_end,
                "timestamp": format_timestamp(current_start),
            })
            # Keep last 5 entries as overlap for the next chunk
            overlap = current_texts[-5:]
            current_texts = overlap
            overlap_count = len(overlap)
            current_start = transcript[i - overlap_count + 1]["start"] if overlap_count > 1 else current_end

    if current_texts:
        chunks.append({
            "text": " ".join(current_texts),
            "start": current_start,
            "end": current_end,
            "timestamp": format_timestamp(current_start),
        })

    return chunks


def _reconstruct_sentences(transcript: list) -> list[dict]:
    """
    Phase 1: Merge all raw caption fragments into complete sentences.

    YouTube captions split mid-sentence ("how did Slack launch their" / "product
    to their initial users?"). This joins everything into continuous text, splits
    on actual sentence-ending punctuation (.?!), and maps each sentence back to
    its start/end timestamps via a character-offset index.

    Returns a list of {"text": str, "start": float, "end": float}.
    """
    import re

    # Build merged text + character-offset → timestamp mapping
    merged_parts = []
    offset_map = []  # (char_start, char_end, time_start, time_end)
    cursor = 0

    for entry in transcript:
        cleaned = _clean_text(entry["text"])
        if not cleaned:
            continue
        if merged_parts:
            merged_parts.append(" ")
            cursor += 1
        char_start = cursor
        merged_parts.append(cleaned)
        cursor += len(cleaned)
        char_end = cursor
        offset_map.append((
            char_start,
            char_end,
            entry["start"],
            entry["start"] + entry["duration"],
        ))

    merged_text = "".join(merged_parts)

    # Split on sentence boundaries: . ? ! followed by space or end-of-string
    sentence_spans = []
    for match in re.finditer(r'[^.?!]*[.?!]+(?:\s|$)|[^.?!]+$', merged_text):
        text = match.group().strip()
        if text:
            sentence_spans.append((match.start(), match.end(), text))

    # Map each sentence span back to timestamps
    sentences = []
    for span_start, span_end, text in sentence_spans:
        sent_time_start = None
        sent_time_end = None
        for off_start, off_end, t_start, t_end in offset_map:
            # Does this entry overlap with the sentence span?
            if off_end > span_start and off_start < span_end:
                if sent_time_start is None:
                    sent_time_start = t_start
                sent_time_end = t_end
        if sent_time_start is not None:
            sentences.append({
                "text": text,
                "start": sent_time_start,
                "end": sent_time_end,
            })

    return sentences


def semantic_chunk_transcript(transcript: list, min_chunk_duration: int = 20) -> list[dict]:
    """
    Semantic chunking on reconstructed sentences.

    Pipeline:
      Phase 1 — Reconstruct full sentences from fragmented captions.
      Phase 2 — Embed all sentences, detect topic boundaries via
                similarity drops between adjacent sentences.
      Phase 3 — Merge sentences between boundaries into final chunks,
                respecting min_chunk_duration.
    """
    import math

    # --- Phase 1: reconstruct sentences ---
    sentences = _reconstruct_sentences(transcript)

    if len(sentences) <= 1:
        return [{
            "text": sentences[0]["text"] if sentences else "",
            "start": sentences[0]["start"] if sentences else 0,
            "end": sentences[0]["end"] if sentences else 0,
            "timestamp": format_timestamp(sentences[0]["start"] if sentences else 0),
        }]

    # --- Phase 2: embed sentences + compute similarities ---
    texts = [s["text"] for s in sentences]
    response = openai_client.embeddings.create(input=texts, model="text-embedding-3-small")
    vectors = [item.embedding for item in response.data]

    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    similarities = [
        cosine_similarity(vectors[i], vectors[i + 1])
        for i in range(len(vectors) - 1)
    ]

    # Adaptive threshold
    mean_sim = sum(similarities) / len(similarities)
    variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
    std_sim = math.sqrt(variance)
    threshold = mean_sim - 0.5 * std_sim

    print(f"Semantic chunking: {len(sentences)} sentences, "
          f"similarity mean={mean_sim:.3f} std={std_sim:.3f} threshold={threshold:.3f}")

    # --- Phase 3: merge sentences into chunks at topic boundaries ---
    chunks = []
    group_texts = [sentences[0]["text"]]
    group_start = sentences[0]["start"]
    group_end = sentences[0]["end"]

    for i, sim in enumerate(similarities):
        sent = sentences[i + 1]
        duration_so_far = group_end - group_start
        is_boundary = sim < threshold and duration_so_far >= min_chunk_duration

        if is_boundary:
            chunks.append({
                "text": " ".join(group_texts),
                "start": group_start,
                "end": group_end,
                "timestamp": format_timestamp(group_start),
            })
            group_texts = [sent["text"]]
            group_start = sent["start"]
            group_end = sent["end"]
        else:
            group_texts.append(sent["text"])
            group_end = sent["end"]

    if group_texts:
        chunks.append({
            "text": " ".join(group_texts),
            "start": group_start,
            "end": group_end,
            "timestamp": format_timestamp(group_start),
        })

    return chunks


def embed_and_store(video_id: str, transcript: list) -> None:
    """
    Chunks the transcript, creates OpenAI embeddings in a single batch call,
    and stores everything in a per-video ChromaDB collection.
    No-ops if the collection already has data (cache hit).
    """
    collection = chroma_client.get_or_create_collection(
        name=f"video_{video_id}",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        print(f"ChromaDB: video {video_id} already indexed, skipping")
        return

    chunks = semantic_chunk_transcript(transcript)
    print(f"ChromaDB: created {len(chunks)} chunks for video {video_id}")

    texts = [chunk["text"] for chunk in chunks]
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    embeddings = [item.embedding for item in response.data]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"{video_id}_{i}" for i in range(len(chunks))],
        metadatas=[{
            "start": chunk["start"],
            "end": chunk["end"],
            "timestamp": chunk["timestamp"],
        } for chunk in chunks],
    )

    print(f"ChromaDB: stored {len(chunks)} chunks for video {video_id}")
