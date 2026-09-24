"""
================================================================================
Document Ingestion Module - PDF Loading Pipeline
================================================================================

This module handles reading and loading raw PDF literature files from the local
corpus directory into structured LangChain Document objects, attaching source
metadata for end-to-end traceability.
================================================================================
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


BOOKS_DIR = Path("data/books")


def load_pdf(file_path: Path) -> List[Document]:
    """
    Loads an individual PDF document using PyPDFLoader.

    Args:
        file_path (Path): Path to the target PDF file.

    Returns:
        List[Document]: List of LangChain Document objects, one per page.
    """
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_library() -> List[Document]:
    """
    Scans the BOOKS_DIR directory and loads all PDF documents into memory.
    Enriches each page document's metadata with the filename (`book_name`)
    to maintain source provenance across chunking, indexing, and retrieval.

    Returns:
        List[Document]: Combined list of page-level Document objects across all books.
    """
    documents: List[Document] = []

    pdf_files = list(BOOKS_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        pdf_documents = load_pdf(pdf_file)

        # Tag each document with the source book filename
        for document in pdf_documents:
            document.metadata["book_name"] = pdf_file.name

        documents.extend(pdf_documents)

    return documents


if __name__ == "__main__":
    documents = load_library()

    print(f"\nTotal pages loaded: {len(documents)}")

    if documents:
        print("\n--- FIRST DOCUMENT SAMPLE ---")
        print(documents[0].page_content[:500])

        print("\n--- METADATA ---")
        print(documents[0].metadata)