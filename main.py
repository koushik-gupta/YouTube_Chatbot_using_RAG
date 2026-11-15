from dotenv import load_dotenv

from chain import create_chain
from llm import create_model
from prompts import create_prompt
from retriever import create_retriever
from transcript import fetch_transcript, split_transcript
from youtube import get_video_id


def main():
    """Entry point for the CLI version of the chatbot.

    Prompts the user for a YouTube URL, builds the RAG chain, then enters an
    interactive loop accepting questions until the user types 'exit'.
    """
    load_dotenv()

    video_url = input("Enter the YouTube video URL: ")
    video_id = get_video_id(video_url)

    transcript = fetch_transcript(video_id)

    if not transcript:
        return

    chunks = split_transcript(transcript)
    model = create_model()
    retriever = create_retriever(chunks, model, video_id)
    prompt = create_prompt()
    chain = create_chain(retriever, prompt, model)

    while True:
        question = input("Enter your question about the video or type exit: ")

        if question.lower() == "exit":
            break

        response = chain.invoke(question)
        print(response)


if __name__ == "__main__":
    main()
