from youtube_transcript_api import YouTubeTranscriptApi

ytt_api = YouTubeTranscriptApi()

fetched_transcript = ytt_api.fetch("UrsmFxEIp5k")

# transcript_list = ytt_api.list(video_id="axCreWC6AHw")

# print(transcript_list)
# is iterable
for snippet in fetched_transcript.to_raw_data():
    print(snippet)

# indexable
last_snippet = fetched_transcript[-1]

# provides a length
snippet_count = len(fetched_transcript)