from functools import lru_cache
from pathlib import Path
import re

from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_safe_video_id(video_id):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", video_id)


def get_collection_name(video_id):
    return f"youtube_{get_safe_video_id(video_id)}"


def create_vector_store(chunks, embeddings, video_id):
    persist_directory = Path(CHROMA_DB_DIR) / get_safe_video_id(video_id)
    collection_name = get_collection_name(video_id)

    if persist_directory.exists():
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )

    return Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )


def create_retriever(chunks, model, video_id):
    embeddings = get_embeddings()
    vector_store = create_vector_store(chunks, embeddings, video_id)
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 12,
            "lambda_mult": 0.5,
        },
    )

    compressor = LLMChainExtractor.from_llm(model)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
