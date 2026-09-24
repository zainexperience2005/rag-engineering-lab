"""
================================================================================
Vector Store Module - ChromaDB & OpenAI Embeddings Integration
================================================================================

This module encapsulates vector database operations:
  1. Instantiating OpenAI embedding model (`text-embedding-3-small`).
  2. Ingesting and persisting chunk embeddings into ChromaDB.
  3. Loading existing persistent vector store for similarity retrieval.
================================================================================
"""

from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

from src.ingestion import load_library
from src.chunking import split_documents

# Load environment variables (such as OPENAI_API_KEY) from .env
load_dotenv()

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "rag_library"


def get_embedding_model() -> OpenAIEmbeddings:
    """
    Initializes and returns the OpenAI embeddings client.
    Uses 'text-embedding-3-small' (1536-dimensional dense vectors) offering
    an optimal balance between speed, cost, and semantic retrieval accuracy.

    Returns:
        OpenAIEmbeddings: Configured LangChain embedding model instance.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Creates or populates a persistent Chroma vector store with document chunks.
    Uses deterministic `chunk_id` values as vector IDs to prevent duplicate entries.

    Args:
        chunks (List[Document]): The chunked documents to embed and store.

    Returns:
        Chroma: Initialized and populated Chroma vector store instance.
    """
    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=ids,
    )

    return vector_store


def load_vector_store() -> Chroma:
    """
    Loads an existing persistent Chroma vector store from disk.

    Returns:
        Chroma: Persistent Chroma vector store instance configured for querying.
    """
    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )


if __name__ == "__main__":
    # Ingestion & Indexing Pipeline Execution
    print("Loading raw corpus documents...")
    documents = load_library()

    print("Splitting documents into chunks...")
    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    print(f"Embedding and persisting {len(chunks)} chunks into ChromaDB at '{CHROMA_PATH}'...")
    vector_store = create_vector_store(chunks)
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB.")

    # Demonstration Similarity Search
    query = "What is retrieval augmented generation?"
    print(f"\nPerforming sample similarity search for: '{query}'")
    results = vector_store.similarity_search_with_score(query=query, k=3)

    for index, (document, score) in enumerate(results, start=1):
        print(f"\n--- RESULT {index} (Distance Score: {score:.4f}) ---")
        print(document.page_content[:300] + "...")
        print("Metadata:", document.metadata)