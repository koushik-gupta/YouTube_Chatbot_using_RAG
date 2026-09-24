from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from youtube_transcript_api import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from config import TRANSCRIPT_LANGUAGES


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class TranscriptFetchError(Exception):
    """A short, user-facing explanation for a transcript retrieval failure."""


def fetch_transcript(video_id):
    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES)
        return [
            {
                "text": chunk.text,
                "start": getattr(chunk, "start", None),
                "duration": getattr(chunk, "duration", None),
                "video_id": video_id,
            }
            for chunk in transcript_list
        ]
    except (TranscriptsDisabled, NoTranscriptFound):
        raise TranscriptFetchError(
            "This video does not have captions in the supported languages. "
            "Try another video or enable captions on the video."
        )
    except VideoUnavailable:
        raise TranscriptFetchError(
            "This video is unavailable, private, or restricted in your region."
        )
    except RequestBlocked:
        raise TranscriptFetchError(
            "YouTube is temporarily blocking transcript requests from your network. "
            "Wait and try again, reduce repeated requests, or use a different network."
        )


def split_transcript(transcript):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if isinstance(transcript, str):
        return text_splitter.create_documents([transcript])

    grouped_documents = []
    documents = []
    current_segments = []
    current_length = 0

    def create_document(segments, chunk_number):
        first_segment = segments[0]
        last_segment = segments[-1]
        return Document(
            page_content=" ".join(segment["text"] for segment in segments),
            metadata={
                "video_id": first_segment.get("video_id"),
                "chunk": chunk_number,
                "segment_start": first_segment["segment"],
                "segment_end": last_segment["segment"],
                "start": first_segment.get("start"),
                "end": last_segment.get("end"),
            },
        )

    def get_overlap_segments(segments):
        overlap_segments = []
        overlap_length = 0

        for segment in reversed(segments):
            overlap_segments.insert(0, segment)
            overlap_length += len(segment["text"])

            if overlap_length >= CHUNK_OVERLAP:
                break

        return overlap_segments

    for index, chunk in enumerate(transcript):
        start = chunk.get("start")
        duration = chunk.get("duration") or 0
        end = start + duration if start is not None else None
        text = chunk["text"]

        segment = {
            "text": text,
            "video_id": chunk.get("video_id"),
            "segment": index + 1,
            "start": start,
            "end": end,
        }

        if current_segments and current_length + len(text) > CHUNK_SIZE:
            grouped_documents.append(create_document(current_segments, len(grouped_documents) + 1))
            current_segments = get_overlap_segments(current_segments)
            current_length = sum(len(segment["text"]) for segment in current_segments)

        current_segments.append(segment)
        current_length += len(text)

    if current_segments:
        grouped_documents.append(create_document(current_segments, len(grouped_documents) + 1))

    documents = text_splitter.split_documents(grouped_documents)
    return documents
