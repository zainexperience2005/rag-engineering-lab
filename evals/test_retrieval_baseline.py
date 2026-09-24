"""
================================================================================
RAG Engineering Lab - Retrieval Baseline Evaluation Harness
================================================================================

This module performs automated, objective evaluation of the Retrieval component
in a Retrieval-Augmented Generation (RAG) system using DeepEval.

Why Evaluate Retrieval Separately from Generation?
---------------------------------------------------
A standard RAG pipeline comprises two distinct stages:
  1. Retrieval: Query -> Vector DB Similarity Search -> Top-k Context Chunks
  2. Generation: Prompt(Query + Context Chunks) -> Generator LLM -> Response

If the retriever fetches irrelevant, incomplete, or out-of-order chunks, the
generation stage will inevitably suffer from:
  - Hallucinations (when missing critical facts)
  - Distraction / Confabulation (due to context noise)
  - Incomplete answers (due to low recall)

By isolating and benchmarking the retrieval stage against a curated ground-truth
dataset, engineers can systematically test and tune chunking strategies, embedding
models, search parameters (e.g., top-k, similarity thresholds), and re-ranking algorithms.

================================================================================
DeepEval Metrics Evaluated
================================================================================

1. Contextual Relevancy (Noise-to-Signal Ratio)
-----------------------------------------------
* DEFINITION:
  Measures the proportion of sentences within the retrieved context that are
  directly relevant to answering the user query.
* FORMULA:
  Relevancy = (Number of Relevant Sentences in Context) / (Total Sentences in Context)
* HOW IT WORKS:
  DeepEval uses an evaluator LLM to parse the retrieved context into individual
  sentences and determines whether each sentence provides direct value in answering
  the input question.
* WHY IT MATTERS:
  Retrieving verbose chunks loaded with boilerplate, headers, or off-topic information
  dilutes the attention of the generator LLM, inflates token costs, and increases the
  likelihood of hallucinations or off-target replies.
* TARGET THRESHOLD:
  >= 0.70 (meaning at least 70% of retrieved context content directly supports the query).

2. Contextual Precision (Rank-Aware Search Quality)
---------------------------------------------------
* DEFINITION:
  Measures whether the most relevant context chunks are ranked higher in the
  retrieval list than less relevant or noisy chunks.
* FORMULA:
  Derived from Mean Average Precision (MAP) @ k:
  Precision@k = (Relevant chunks among top k) / k
  Contextual Precision = Sum(Precision@k * Relevance_indicator_k) / (Total Relevant Chunks)
* HOW IT WORKS:
  The evaluator LLM assesses each retrieved chunk in order of appearance (rank 1, 2, ... k)
  against the expected ground-truth answer. Higher scores are awarded when relevant nodes
  appear at rank 1 rather than being buried at rank 3 or 5.
* WHY IT MATTERS:
  LLMs exhibit the well-documented "Lost in the Middle" phenomenon (Liu et al.):
  information positioned at the very beginning of the context receives the strongest
  attention weight. If the best answer chunk is ranked last, the generator may overlook it.
* TARGET THRESHOLD:
  >= 0.70 (verifying top-ranked context items are high-confidence relevant matches).

3. Contextual Recall (Completeness of Information)
--------------------------------------------------
* DEFINITION:
  Measures whether the retrieved context contains all the necessary facts and
  evidence needed to produce the ground-truth expected answer.
* FORMULA:
  Recall = (Number of Ground Truth Statements Attributable to Retrieved Context) /
           (Total Statements in Expected Answer)
* HOW IT WORKS:
  The evaluator LLM extracts key factual claims and assertions from the `expected_output`
  and verifies whether each claim can be strictly deduced from the `retrieval_context`.
* WHY IT MATTERS:
  If retrieval recall is low, the generator LLM either fails to answer or is forced
  to rely on its pre-trained parametric knowledge (which introduces risks of stale facts
  or hallucinations).
* TARGET THRESHOLD:
  >= 0.70 (meaning at least 70% of the ground-truth facts are present in retrieved chunks).
================================================================================
"""

from typing import List
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from evals.dataset import EVALUATION_DATASET
from src.retriever import retrieve


def build_test_cases() -> List[LLMTestCase]:
    """
    Constructs a list of DeepEval test cases from the curated evaluation dataset.

    For each question in `EVALUATION_DATASET`:
      1. Executes vector similarity search via `src.retriever.retrieve(query, k=3)`.
      2. Extracts raw text chunks into `retrieval_context`.
      3. Packages `input` (question), `expected_output` (ground truth), and
         `retrieval_context` into an `LLMTestCase`.
      4. Sets `actual_output` to `expected_answer` because we are strictly
         evaluating the retrieval layer (context relevance, ranking, and completeness)
         in isolation before feeding into an LLM generator.

    Returns:
        List[LLMTestCase]: Ready-to-evaluate test cases for DeepEval.
    """
    test_cases: List[LLMTestCase] = []

    for item in EVALUATION_DATASET:
        # Retrieve the top-k document chunks from the Chroma vector database
        documents = retrieve(
            query=item["question"],
            k=3,
        )

        # Extract textual content from LangChain Document objects
        retrieval_context = [
            document.page_content
            for document in documents
        ]

        # Assemble the DeepEval test case
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["expected_answer"],
            expected_output=item["expected_answer"],
            retrieval_context=retrieval_context,
        )

        test_cases.append(test_case)

    return test_cases


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Starting RAG Retrieval Baseline Evaluation")
    print("=" * 80)

    # 1. Build test cases by executing live retrieval across the corpus
    test_cases = build_test_cases()
    print(f"Constructed {len(test_cases)} test cases from evaluation dataset.\n")

    # 2. Configure DeepEval evaluation metrics with production-grade 0.70 thresholds
    metrics = [
        ContextualRelevancyMetric(
            threshold=0.7,
            # Evaluates sentence-level noise in retrieved chunks
        ),
        ContextualPrecisionMetric(
            threshold=0.7,
            # Evaluates whether the best chunks are ranked at the top
        ),
        ContextualRecallMetric(
            threshold=0.7,
            # Evaluates whether all ground-truth facts are captured in context
        ),
    ]

    # 3. Execute DeepEval evaluation suite
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )