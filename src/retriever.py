"""
================================================================================
Retriever Module - Vector Similarity Query Interface
================================================================================

This module exposes clean interfaces to retrieve top-k semantically relevant
document chunks from ChromaDB for a given natural language query.
================================================================================
"""

from typing import List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.vectorstore import load_vector_store


def get_retriever(k: int = 3) -> VectorStoreRetriever:
    """
    Constructs a LangChain VectorStoreRetriever instance configured for
    similarity-based nearest-neighbor search.

    Args:
        k (int, optional): Number of top document chunks to retrieve. Defaults to 3.

    Returns:
        VectorStoreRetriever: Configured retriever object.
    """
    vector_store = load_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )


def retrieve(query: str, k: int = 3) -> List[Document]:
    """
    Executes a vector similarity query against the persistent Chroma vector store.

    Args:
        query (str): The search query or question.
        k (int, optional): The number of top chunks to return. Defaults to 3.

    Returns:
        List[Document]: The top-k most relevant Document chunks.
    """
    retriever = get_retriever(k=k)
    documents = retriever.invoke(query)
    return documents


if __name__ == "__main__":
    question = "What is corrective RAG?"

    print(f"\nQuerying: '{question}' (top k=3)")
    documents = retrieve(
        query=question,
        k=3,
    )

    for index, document in enumerate(documents, start=1):
        print(f"\n--- RETRIEVED DOCUMENT {index} ---")
        print(document.page_content.strip())
        print("\nMetadata:")
        print(document.metadata)