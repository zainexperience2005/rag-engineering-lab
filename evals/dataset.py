"""
================================================================================
RAG Evaluation Dataset - Ground Truth Benchmark Suite
================================================================================

This module defines the golden test dataset used to evaluate the retrieval
and generation quality of the RAG system.

Evaluation Dataset Design Principles:
-------------------------------------
1. Objective Ground Truth:
   Each question is paired with a strictly verified `expected_answer`. This
   ground-truth text serves as the reference against which:
     - Contextual Recall checks if the retriever surfaced all essential facts.
     - Contextual Precision checks if top-ranked chunks align with the target answer.
     - Generator correctness / faithfulness can be verified.

2. Diverse Query Typology:
   The dataset covers a multi-faceted spectrum of information retrieval challenges:
     - Fundamental Concept Definitions (e.g., embeddings, atomicity).
     - Specialized Architectural Patterns (e.g., Corrective RAG).
     - RAG Evaluation Metrics (Contextual Precision, Contextual Recall).
     - Observability & Systems Engineering (Tracing vs. Evaluation).
     - Hyperparameter Trade-offs (Impact of large top-k).
     - Needle-in-a-Haystack / Code Name Retrieval (e.g., ORBIT-LANTERN-42) to test
       exact keyword and specific entity retrieval against corpus distractors.

3. Regression Prevention:
   Running this dataset after modifications to chunk size, chunk overlap,
   embedding models, or reranking steps ensures that improvements in one area
   do not degrade retrieval precision or recall in another.
================================================================================
"""

from typing import List, Dict

EVALUATION_DATASET: List[Dict[str, str]] = [
    # --------------------------------------------------------------------------
    # Test Case 1: Database Fundamentals (Atomicity)
    # Purpose: Tests domain knowledge retrieval on ACID transactions.
    # --------------------------------------------------------------------------
    {
        "question": "What does atomicity mean in a database transaction?",
        "expected_answer": (
            "Atomicity means a transaction's changes succeed "
            "as a unit or are rolled back."
        ),
    },

    # --------------------------------------------------------------------------
    # Test Case 2: Machine Learning & NLP Core Concepts (Embeddings)
    # Purpose: Tests retrieval of conceptual definitions regarding geometric vector spaces.
    # --------------------------------------------------------------------------
    {
        "question": "What is an embedding?",
        "expected_answer": (
            "An embedding is a dense numeric representation "
            "that allows useful relationships to be represented geometrically."
        ),
    },

    # --------------------------------------------------------------------------
    # Test Case 3: Advanced RAG Architecture (Corrective RAG)
    # Purpose: Evaluates retrieval of workflow-specific agentic RAG mechanisms
    # (query rewrites, fallbacks, web search triggers).
    # --------------------------------------------------------------------------
    {
        "question": "What is corrective RAG?",
        "expected_answer": (
            "Corrective RAG assesses retrieved evidence and can "
            "rewrite the query, retrieve again, or use another source "
            "when the evidence is inadequate."
        ),
    },

    # --------------------------------------------------------------------------
    # Test Case 4: Evaluation Metric Definition (Contextual Precision)
    # Purpose: Tests system self-knowledge on retrieval ranking and signal focus.
    # --------------------------------------------------------------------------
    {
        "question": "What is contextual precision in RAG?",
        "expected_answer": (
            "Contextual precision measures whether retrieved "
            "context is focused on relevant information."
        ),
    },

    # --------------------------------------------------------------------------
    # Test Case 5: Evaluation Metric Definition (Contextual Recall)
    # Purpose: Tests system understanding of ground-truth fact coverage in context.
    # --------------------------------------------------------------------------
    {
        "question": "What is contextual recall in RAG?",
        "expected_answer": (
            "Contextual recall measures whether the retrieved "
            "context contains the information needed to answer "
            "the question."
        ),
    },

    # --------------------------------------------------------------------------
    # Test Case 6: Systems Engineering & Observability (Tracing vs. Evaluation)
    # Purpose: Tests fine-grained conceptual differentiation between runtime telemetry
    # (tracing) and qualitative measurement (evaluation).
    # --------------------------------------------------------------------------
    {
        "question": "What is the difference between tracing and evaluation?",
        "expected_answer": (
            "Tracing records what happened during execution, "
            "while evaluation assigns a measurement or judgment "
            "to that behavior."
        ),
    },

    # --------------------------------------------------------------------------
    # Test Case 7: RAG Trade-offs & Engineering Hyperparameters (Top-k tuning)
    # Purpose: Tests knowledge regarding signal-to-noise ratio degradation
    # when top-k is over-expanded.
    # --------------------------------------------------------------------------
    {
        "question": "Why can a large top-k value hurt a RAG system?",
        "expected_answer": (
            "A large top-k can introduce irrelevant context "
            "even though it may improve recall."
        ),
    },

    # --------------------------------------------------------------------------
    # Test Case 8: Needle-in-a-Haystack Specific Entity (Corpus Project Code Name)
    # Purpose: Tests retrieval resilience against controlled distractors and
    # exact identifier location within multi-book corpora.
    # --------------------------------------------------------------------------
    {
        "question": "What is the corpus project code name?",
        "expected_answer": "ORBIT-LANTERN-42.",
    },
]