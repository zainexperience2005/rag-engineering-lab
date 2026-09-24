"""
================================================================================
RAG Engineering Lab - End-to-End System Evaluation Harness
================================================================================

This module executes a holistic, end-to-end evaluation of the compiled LangGraph
RAG pipeline against the curated golden benchmark dataset using DeepEval.

Why End-to-End Evaluation?
--------------------------
While unit evaluations isolate individual components:
  - `test_retrieval_baseline.py` isolates the Retriever (Contextual Relevancy, Precision, Recall)
  - `test_generator_baseline.py` isolates the Generator (Answer Relevancy, Faithfulness)

The end-to-end test evaluates the full synthesis:
How effectively does the entire pipeline (`Query -> Graph -> Context -> Answer`)
perform as a cohesive system in production?

================================================================================
The 5 Core RAG Evaluation Metrics
================================================================================

1. Contextual Relevancy (Noise Filter Quality)
----------------------------------------------
* DEFINITION:
  Measures the fraction of sentences in the retrieved context that are relevant
  to answering the input question.
* FORMULA:
  Contextual Relevancy = (Number of Relevant Sentences in Context) / (Total Sentences in Context)
* WORKING:
  Evaluator LLM parses the retrieved chunks into individual sentences and counts
  how many directly contribute toward answering the query vs. off-target noise.

2. Contextual Precision (Rank Quality)
--------------------------------------
* DEFINITION:
  Measures whether the most relevant context chunks are ranked higher (positions 1, 2)
  rather than being placed after distracting or irrelevant chunks.
* FORMULA:
  Mean Average Precision (MAP) @ k. Higher weight is given when relevant nodes appear first.
* WORKING:
  Evaluates each retrieved chunk in order of retrieval rank against the target answer,
  heavily penalizing relevant information placed far down the prompt ("Lost in the Middle").

3. Contextual Recall (Factual Completeness)
-------------------------------------------
* DEFINITION:
  Measures whether the retrieved context contains all the necessary facts and
  statements present in the golden ground-truth expected answer.
* FORMULA:
  Contextual Recall = (Ground Truth Statements Attributable to Context) /
                      (Total Statements in Expected Answer)
* WORKING:
  Deconstructs the expected answer into atomic factual statements and checks
  whether each claim is supported by the retrieved context.

4. Answer Relevancy (Directness of Response)
--------------------------------------------
* DEFINITION:
  Measures whether the generated answer directly addresses the user query without
  drifting into tangential commentary, unnecessary padding, or evasiveness.
* FORMULA:
  Answer Relevancy = (Relevant Statements in Answer) / (Total Statements in Answer)
* WORKING:
  Evaluates the semantic alignment and utility of every sentence in `actual_output`
  relative to the prompt `input`.

5. Faithfulness (Hallucination Detection)
-----------------------------------------
* DEFINITION:
  Measures whether every factual claim in the generated answer can be strictly
  inferred from the retrieved context, ensuring zero confabulation.
* FORMULA:
  Faithfulness = (Truthful Statements in Answer Supported by Context) /
                 (Total Statements in Answer)
* WORKING:
  Extracts all factual assertions from `actual_output` and cross-verifies each
  assertion against the provided `retrieval_context`.
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
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from evals.dataset import EVALUATION_DATASET
from src.graph import rag_graph


def build_test_cases() -> List[LLMTestCase]:
    """
    Constructs a list of LLMTestCase instances by executing the full LangGraph RAG pipeline.

    For each question in `EVALUATION_DATASET`:
      1. Invokes `rag_graph.invoke(...)` executing both retrieve_node and generate_node.
      2. Extracts the synthesized answer and the retrieved document chunks.
      3. Packages the inputs, outputs, ground truth, and context into an `LLMTestCase`.

    Returns:
        List[LLMTestCase]: Complete list of test cases ready for DeepEval evaluation.
    """
    test_cases: List[LLMTestCase] = []

    for item in EVALUATION_DATASET:
        # Execute the compiled LangGraph workflow end-to-end
        result = rag_graph.invoke(
            {
                "question": item["question"],
                "documents": [],
                "answer": "",
            }
        )

        # Extract textual page contents from retrieved documents
        retrieval_context = [
            document.page_content
            for document in result["documents"]
        ]

        # Assemble the test case
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=result["answer"],
            expected_output=item["expected_answer"],
            retrieval_context=retrieval_context,
        )

        test_cases.append(test_case)

    return test_cases


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Starting End-to-End RAG Pipeline Evaluation")
    print("=" * 80)

    # 1. Execute end-to-end pipeline across benchmark dataset
    test_cases = build_test_cases()
    print(f"Generated {len(test_cases)} end-to-end test cases.\n")

    # 2. Configure 5 production evaluation metrics
    metrics = [
        # Generation Metrics
        AnswerRelevancyMetric(
            threshold=0.7,
            # Evaluates whether the generated response directly answers the query
        ),
        FaithfulnessMetric(
            threshold=0.7,
            # Evaluates whether the response is strictly grounded in retrieved evidence
        ),

        # Retrieval Metrics
        ContextualRelevancyMetric(
            threshold=0.7,
            # Evaluates signal-to-noise ratio in retrieved context chunks
        ),
        ContextualPrecisionMetric(
            threshold=0.7,
            # Evaluates whether top-ranked chunks are the most relevant
        ),
        ContextualRecallMetric(
            threshold=0.7,
            # Evaluates whether retrieved context captures all required facts
        ),
    ]

    # 3. Run DeepEval evaluation
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )