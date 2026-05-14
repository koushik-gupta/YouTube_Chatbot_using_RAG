from langchain_core.prompts import PromptTemplate


def create_prompt():
    return PromptTemplate(
        template="""
        your first task is to understand the full context and rectify any wrongly transcribed words. then you have to answer the question based on the provided context. if the answer is not present in the context then you have to say that you don't know. do not try to answer the question if the answer is not present in the context. also do not try to make up an answer. just say that you don't know if the answer is not present in the context.

        also if the cantent is not in english then you have to translate it to english first and then answer the question. if the content is in english then you can directly answer the question without translating it.
      You are a helpful assistant.


      {context}
      Question: {question}
    """,
        input_variables=["context", "question"],
    )
