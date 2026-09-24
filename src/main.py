"""
================================================================================
RAG Engineering Lab - Interactive Command-Line Assistant
================================================================================

This module provides a command-line interface (CLI) to interactively query
the LangGraph RAG pipeline and inspect both the generated answers and the
underlying source citations retrieved from the vector store.
================================================================================
"""

import sys
from src.graph import rag_graph


def main() -> None:
    """
    Runs the main interactive loop for the RAG Assistant.
    Prompts the user for queries, executes the compiled LangGraph workflow,
    and displays the generated response alongside source document citations.
    """
    # Ensure Windows console supports unicode symbols without error
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    print("\n" + "=" * 60)
    print("🤖  RAG Engineering Lab - Interactive Assistant")
    print("=" * 60)
    print("Ask any question based on the indexed document library.")
    print("Type 'exit' or 'quit' to terminate the session.\n")

    while True:
        try:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in ("exit", "quit", "q"):
                print("\nExiting session. Goodbye!")
                break

            # Execute compiled LangGraph pipeline
            result = rag_graph.invoke(
                {
                    "question": question,
                    "documents": [],
                    "answer": "",
                }
            )

            # Display generated answer
            print("\nAssistant:")
            print(result["answer"])

            # Display source provenance
            documents = result.get("documents", [])
            if documents:
                print("\nSources:")
                for index, document in enumerate(documents, start=1):
                    book_name = document.metadata.get("book_name", "Unknown Book")
                    page = document.metadata.get("page", "Unknown")
                    chunk_id = document.metadata.get("chunk_id", "N/A")
                    print(f"  [{index}] {book_name} (Page {page}) - ID: {chunk_id}")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession interrupted by user. Goodbye!")
            break
        except Exception as error:
            print(f"\n[Error executing query]: {error}")


if __name__ == "__main__":
    main()