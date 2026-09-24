"""
================================================================================
LangGraph RAG Workflow - StateGraph Definition
================================================================================

This module implements a production-grade, graph-based Retrieval-Augmented
Generation (RAG) pipeline using LangGraph and LangChain.

Workflow Architecture:
----------------------
           [START]
              │
              ▼
       ┌──────────────┐
       │ retrieve_node│  -> Queries ChromaDB for top-k document chunks
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │ generate_node│  -> Synthesizes grounded answer using strict system prompt
       └──────┬───────┘
              │
              ▼
            [END]

Design Principles:
------------------
1. Strict Grounding:
   The generator prompt enforces zero reliance on external parametric knowledge
   to eliminate hallucinations. If context is insufficient, it yields a standardized refusal.
2. State Immutability:
   State transitions are managed through a typed dictionary (`RAGState`), ensuring
   deterministic tracing, debugging, and reproducibility.
================================================================================
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from src.retriever import retrieve

# Load environment variables
load_dotenv()


class RAGState(TypedDict):
    """
    Defines the shared state schema flowing across nodes in the LangGraph RAG pipeline.

    Attributes:
        question (str): The natural language query from the user.
        documents (List[Document]): List of retrieved document chunks used as factual context.
        answer (str): The final synthesized response produced by the LLM.
    """
    question: str
    documents: List[Document]
    answer: str


# Initialize LLM with zero temperature for deterministic, fact-grounded synthesis
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    temperature=0,
)


def retrieve_node(state: RAGState) -> dict:
    """
    Retrieval Node:
    Extracts the user question from state and fetches the top-k most semantically
    relevant document chunks from the Chroma vector database.

    Args:
        state (RAGState): Current graph state containing the user question.

    Returns:
        dict: State update dictionary containing retrieved `documents`.
    """
    question = state["question"]

    documents = retrieve(
        query=question,
        k=3,
    )

    return {
        "documents": documents
    }


def generate_node(state: RAGState) -> dict:
    """
    Generation Node:
    Formats retrieved documents into a structured context window and prompts
    the LLM to generate an answer strictly grounded in the provided evidence.

    Args:
        state (RAGState): Current graph state containing `question` and `documents`.

    Returns:
        dict: State update dictionary containing the synthesized `answer`.
    """
    question = state["question"]
    documents = state.get("documents", [])

    # Handle scenario where no documents were retrieved
    if not documents:
        return {
            "answer": "I don't have enough information in the provided documents."
        }

    # Aggregate context passages separated by double newlines
    context = "\n\n".join(
        document.page_content.strip()
        for document in documents
    )

    system_prompt = """You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information, say: "I don't have enough information in the provided documents."
3. Do not invent facts or extrapolate beyond what is explicitly stated.
4. Keep the answer concise, accurate, and clear."""

    user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context}"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return {
        "answer": response.content
    }


# ------------------------------------------------------------------------------
# Graph Construction & Compilation
# ------------------------------------------------------------------------------

builder = StateGraph(RAGState)

# Register functional nodes
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)

# Define deterministic execution flow
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

# Compile into an executable graph instance
rag_graph = builder.compile()