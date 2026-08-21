import os

from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def create_groq_model():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.7,
    )


def create_huggingface_model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        provider="novita",
        task="conversational",
        temperature=0.7,
    )
    return ChatHuggingFace(llm=llm)


def create_model():
    if os.getenv("GROQ_API_KEY"):
        return create_groq_model()

    return create_huggingface_model()
