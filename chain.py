from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


def _seconds_to_timestamp(seconds):
    """Convert raw seconds (float/int) to a MM:SS or HH:MM:SS string."""
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_docs(retrieved_docs):
    """Format a list of retrieved Document objects into a single annotated string.

    Each document is prefixed with a bracketed metadata header containing the
    video ID, start/end timestamps, chunk number, and segment range (when available).
    Timestamps are pre-formatted as MM:SS so the LLM cites them in a readable form.
    The resulting string is passed as the {context} variable into the prompt template.
    """
    formatted_docs = []

    for doc in retrieved_docs:
        metadata = doc.metadata
        source_parts = []

        if metadata.get("video_id"):
            source_parts.append(f"video_id={metadata['video_id']}")

        if metadata.get("start") is not None:
            source_parts.append(f"start={_seconds_to_timestamp(metadata['start'])}")

        if metadata.get("end") is not None:
            source_parts.append(f"end={_seconds_to_timestamp(metadata['end'])}")

        if metadata.get("chunk"):
            source_parts.append(f"chunk={metadata['chunk']}")

        if metadata.get("segment_start") and metadata.get("segment_end"):
            source_parts.append(
                f"segments={metadata['segment_start']}-{metadata['segment_end']}"
            )

        source = ", ".join(source_parts) or "source=unknown"
        formatted_docs.append(f"[{source}]\n{doc.page_content}")

    return "\n\n".join(formatted_docs)


def create_chain(retriever, prompt, model):
    """Build and return the LCEL RAG chain.

    The chain runs the retriever and the passthrough question in parallel,
    fills the prompt template with the results, passes it to the LLM,
    and parses the output into a plain string.
    """
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    parser = StrOutputParser()
    return parallel_chain | prompt | model | parser
