from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def create_model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        provider="novita",
        task="conversational",
        temperature=0.7,
    )
    return ChatHuggingFace(llm=llm)
