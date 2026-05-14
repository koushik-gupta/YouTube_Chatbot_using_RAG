from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

from config import TRANSCRIPT_LANGUAGES


def fetch_transcript(video_id):
    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
        return " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        print("No captions available for this video.")
        return ""


def split_transcript(transcript):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.create_documents([transcript])
