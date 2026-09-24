"""
================================================================================
RAG Engineering Lab - Generator Baseline Evaluation Harness
================================================================================

This module performs isolated, automated evaluation of the **Generation** component
in a Retrieval-Augmented Generation (RAG) system using DeepEval.

Why Evaluate the Generator in Isolation?
----------------------------------------
In an end-to-end RAG system, bad answers can stem from two distinct failure modes:
  1. Retrieval Failure: The retriever missed the required facts (low recall) or
     flooded the prompt with noise (low relevancy).
  2. Generation Failure: The LLM possessed the right facts in the context, but
     hallucinated, extrapolated beyond the text, failed to follow formatting rules,
     or drifted away from the user's specific question.

By isolating the generator evaluation, engineers can evaluate:
  - System prompt compliance (strict refusal vs. guessing).
  - Faithfulness to context (zero hallucinations).
  - Direct relevance of generated answers without unnecessary padding.

================================================================================
DeepEval Metrics Evaluated for Generation
================================================================================

1. Answer Relevancy (Question-to-Answer Directness)
---------------------------------------------------
* DEFINITION:
  Measures whether the generated answer directly addresses the user's question,
  without extraneous fluff, evasiveness, or tangential discourse.
* FORMULA:
  Answer Relevancy = (Number of Relevant Statements in Output) / (Total Statements in Output)
* HOW IT WORKS:
  DeepEval uses an evaluator LLM to extract all atomic statements from the `actual_output`
  and evaluates whether each statement contributes directly toward satisfying the user's
  original `input` prompt.
* WHY IT MATTERS:
  Users want concise, actionable answers. Models that generate polite filler, repetitive
  disclaimers, or unprompted background info receive low relevancy scores.
* TARGET THRESHOLD:
  >= 0.70

2. Faithfulness (Hallucination Detection & Context Grounding)
-------------------------------------------------------------
* DEFINITION:
  Measures whether the generated response is strictly factually grounded in the
  retrieval context, without introducing fabricated or unsupported claims.
* FORMULA:
  Faithfulness = (Number of Supported Truthful Statements in Output) /
                 (Total Factual Statements in Output)
* HOW IT WORKS:
  DeepEval extracts all factual claims made in the `actual_output` and cross-examines
  them against the text in `retrieval_context`. If a claim cannot be verified directly
  from the provided context chunks, it is flagged as an unsupported hallucination.
* WHY IT MATTERS:
  Hallucinations are the #1 barrier to deploying enterprise RAG. High faithfulness
  guarantees that answers reflect trusted institutional knowledge rather than
  speculative parametric weights.
* TARGET THRESHOLD:
  >= 0.70
================================================================================
"""

import sys
from typing import List

# Ensure Windows consoles support unicode / emoji printing without charmap encoding crashes
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)

from evals.dataset import EVALUATION_DATASET
from src.retriever import retrieve
from src.graph import generate_node, RAGState


def build_generator_test_cases() -> List[LLMTestCase]:
    """
    Constructs test cases specifically designed to evaluate the generator node.

    Execution Pipeline:
      1. Retrieves context for each question in `EVALUATION_DATASET`.
      2. Invokes the generator node (`generate_node`) using the question and retrieved context.
      3. Packages `input` (question), `actual_output` (generated answer),
         `expected_output` (ground truth), and `retrieval_context` into an `LLMTestCase`.

    Returns:
        List[LLMTestCase]: Evaluatable test cases focused on generation quality.
    """
    test_cases: List[LLMTestCase] = []

    for item in EVALUATION_DATASET:
        question = item["question"]

        # Retrieve documents to provide as context to the generator
        documents = retrieve(
            query=question,
            k=3,
        )

        retrieval_context = [
            document.page_content
            for document in documents
        ]

        # Execute generator node in isolation
        state: RAGState = {
            "question": question,
            "documents": documents,
            "answer": "",
        }
        generation_result = generate_node(state)
        actual_output = generation_result["answer"]

        # Build test case
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=item["expected_answer"],
            retrieval_context=retrieval_context,
        )

        test_cases.append(test_case)

    return test_cases


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Starting RAG Generator Baseline Evaluation")
    print("=" * 80)

    # 1. Build test cases by executing the generation node
    test_cases = build_generator_test_cases()
    print(f"Constructed {len(test_cases)} generator test cases.\n")

    # 2. Configure DeepEval metrics targeting generation quality
    metrics = [
        AnswerRelevancyMetric(
            threshold=0.7,
            # Evaluates whether the generated response directly answers the query
        ),
        FaithfulnessMetric(
            threshold=0.7,
            # Evaluates whether the generated response contains zero hallucinations
        ),
    ]

    # 3. Execute evaluation
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )
