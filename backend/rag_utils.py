import os
from langchain_openai import ChatOpenAI


def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    if vectorstore is None:
        raise ValueError("A vector store is required for retrieval.")
    results = vectorstore.similarity_search(query, k=k)
    return [f"Content:\n{doc.page_content}\n\nMetadata:\n{doc.metadata}" for doc in results]


def generate(query: str, retrieved_chunks: list[str], model):
    context = "\n\n---\n\n".join(retrieved_chunks)
    messages = [
        ("system", "You are AuroraStays' hotel policy assistant. Answer using only the supplied policy context. If the context does not contain the requested detail, say that the policy documents do not specify it and recommend staff verify with the property. Do not invent fees, deadlines, amenities, or exceptions. Keep the answer concise and cite the source filename when available in metadata."),
        ("human", f"Policy context:\n{context}\n\nGuest question:\n{query}"),
    ]
    return model.invoke(messages)


def rag(query: str, k: int = 2, model_name: str = "gpt-4o-mini", temperature: float = 0.1, top_p: float = 0.9, max_tokens: int = 512, vectorstore=None):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key or not api_base:
        raise RuntimeError("Set OPENAI_API_KEY and OPENAI_API_BASE in the deployment environment.")
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, openai_api_key=api_key, openai_api_base=api_base)
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer.content, retrieved_chunks
