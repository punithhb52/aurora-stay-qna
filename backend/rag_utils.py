from langchain_openai import ChatOpenAI
import os


def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    if vectorstore is None:
        raise ValueError("A vector store is required for retrieval.")
    results = vectorstore.similarity_search(query, k=k)
    return [f"Content:
{doc.page_content}

Metadata:
{doc.metadata}" for doc in results]


def generate(query: str, retrieved_chunks: list[str], model):
    context = "

".join(retrieved_chunks)
    prompt = f"""You are AuroraStays' guest policy assistant. Answer using only the retrieved hotel policy context below.

Guest question:
{query}

Retrieved policy context:
{context}

Guidelines:
- Answer the specific question clearly and concisely.
- Do not guess or use outside knowledge. Preserve property-specific conditions, limits, fees, exceptions, and distinctions.
- If a required detail is absent, say that the retrieved documents do not provide it and explain what must be checked.
- If sources conflict, describe the conflict without guessing.
- Cite supporting source filenames using the source metadata shown in the retrieved context.

Answer:"""
    return model.invoke(prompt)


def rag(query: str, k: int = 2, model_name: str = "gpt-4o-mini", temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 512, vectorstore=None):
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key or not api_base:
        raise RuntimeError("Set OPENAI_API_KEY and OPENAI_API_BASE in the deployment environment.")
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, openai_api_key=api_key, openai_api_base=api_base)
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    answer = generate(query=query, retrieved_chunks=retrieved_chunks, model=model)
    return answer.content, retrieved_chunks
