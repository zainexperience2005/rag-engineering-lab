"""
================================================================================
Document Chunking Module - Recursive Character Text Splitting
================================================================================

This module transforms page-level documents into smaller, semantically coherent
chunks suitable for vector embeddings and similarity search.

Key Concepts:
-------------
- chunk_size (default: 1000 characters): The maximum length of each chunk.
  Larger chunks preserve broader narrative context but increase noise and token cost.
  Smaller chunks isolate specific facts but may break semantic coherence.
- chunk_overlap (default: 200 characters): The sliding window overlap between
  successive chunks. Prevents sentences and crucial context from being cut off at
  chunk boundaries.
- chunk_id: A deterministic identifier formatted as:
  `{book_name}-page-{page}-chunk-{index}`
  ensuring full auditability back to the source page and document.
================================================================================
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion import load_library


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits page-level documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Generates a deterministic unique ID (`chunk_id`) stored in each chunk's metadata.

    Args:
        documents (List[Document]): The raw page documents to split.
        chunk_size (int, optional): Maximum characters per chunk. Defaults to 1000.
        chunk_overlap (int, optional): Character overlap between consecutive chunks. Defaults to 200.

    Returns:
        List[Document]: List of chunked Document objects with unique `chunk_id` metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Attach deterministic chunk identifiers for deduplication and provenance
    for index, chunk in enumerate(chunks):
        book_name = chunk.metadata.get("book_name", "unknown")
        page = chunk.metadata.get("page", "unknown")

        chunk.metadata["chunk_id"] = (
            f"{book_name}-page-{page}-chunk-{index}"
        )

    return chunks


if __name__ == "__main__":
    documents = load_library()
    chunks = split_documents(documents)

    print(f"Original documents/pages: {len(documents)}")
    print(f"Total chunks generated: {len(chunks)}")

    if chunks:
        print("\n--- FIRST CHUNK SAMPLE ---")
        print(chunks[0].page_content)

        print("\n--- METADATA ---")
        print(chunks[0].metadata)