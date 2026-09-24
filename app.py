import re

import streamlit as st
from dotenv import load_dotenv

from chain import create_chain
from llm import create_model
from prompts import create_prompt
from retriever import create_retriever
from transcript import TranscriptFetchError, fetch_transcript, split_transcript
from youtube import get_video_id


TIMESTAMP_RANGE_PATTERN = re.compile(
    r"(?:\[\s*)?(?P<start>\d+(?:\.\d+)?)\s*(?:s|seconds)?\s*(?:-|–|to)\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*(?:s|seconds)\b(?:\s*\])?",
    re.IGNORECASE,
)
TIMESTAMP_PATTERN = re.compile(
    r"(?:\[\s*)?(?<![\w.])(?P<seconds>\d+(?:\.\d+)?)\s*(?:s|seconds)\b(?:\s*\])?",
    re.IGNORECASE,
)


def format_timestamp(seconds):
    total_seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    return f"{minutes}:{seconds:02d}"


def format_answer_timestamps(answer):
    """Turn raw transcript seconds into readable timestamps."""

    def replace_range(match):
        return f"{format_timestamp(match.group('start'))} - {format_timestamp(match.group('end'))}"

    answer = TIMESTAMP_RANGE_PATTERN.sub(replace_range, answer)

    def replace_timestamp(match):
        seconds = match.group("seconds")
        return format_timestamp(seconds)

    return TIMESTAMP_PATTERN.sub(replace_timestamp, answer)


@st.cache_resource
def get_model():
    _bust_cache = True
    return create_model()


@st.cache_resource
def get_chain(video_id):
    _bust_cache = True
    transcript = fetch_transcript(video_id)

    if not transcript:
        return None

    chunks = split_transcript(transcript)
    model = get_model()
    retriever = create_retriever(chunks, model, video_id)
    prompt = create_prompt()
    return create_chain(retriever, prompt, model)


def answer_question(video_url, question):
    video_id = get_video_id(video_url)

    if not video_id:
        return "Please enter a valid YouTube video URL."

    chain = get_chain(video_id)

    answer = chain.invoke(question)
    return format_answer_timestamps(answer)


def main():
    load_dotenv()

    st.set_page_config(
        page_title="YouTube Chatbot",
        layout="centered",
    )

    st.title("YouTube Chatbot")

    with st.form("youtube_chat_form"):
        video_url = st.text_input("YouTube video URL")
        question = st.text_area("Question", height=120)
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not video_url.strip():
            st.warning("Please enter a YouTube video URL.")
            return

        if not question.strip():
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner("Reading transcript and generating answer..."):
                response = answer_question(video_url.strip(), question.strip())
        except TranscriptFetchError as error:
            st.warning(str(error))
            return
        except Exception:
            st.error("The request could not be completed. Please try again shortly.")
            return

        st.subheader("Answer")
        st.write(response)


if __name__ == "__main__":
    main()
