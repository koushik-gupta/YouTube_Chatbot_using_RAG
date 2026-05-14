import streamlit as st
from dotenv import load_dotenv

from chain import create_chain
from llm import create_model
from prompts import create_prompt
from retriever import create_retriever
from transcript import fetch_transcript, split_transcript
from youtube import get_video_id


@st.cache_resource
def get_model():
    return create_model()


@st.cache_resource
def get_chain(video_id):
    transcript = fetch_transcript(video_id)

    if not transcript:
        return None

    chunks = split_transcript(transcript)
    model = get_model()
    retriever = create_retriever(chunks, model)
    prompt = create_prompt()
    return create_chain(retriever, prompt, model)


def answer_question(video_url, question):
    video_id = get_video_id(video_url)

    if not video_id:
        return "Please enter a valid YouTube video URL."

    chain = get_chain(video_id)

    if chain is None:
        return "No captions available for this video."

    return chain.invoke(question)


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
        except Exception as error:
            st.error(f"Something went wrong: {error}")
            return

        st.subheader("Answer")
        st.write(response)


if __name__ == "__main__":
    main()
