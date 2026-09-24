# 🔬 RAG Engineering Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com/)
[![DeepEval](https://img.shields.io/badge/Evals-DeepEval-purple.svg)](https://confident-ai.com/)
[![OpenAI](https://img.shields.io/badge/Models-GPT--4.1--mini%20%7C%20Embedding--3--small-black.svg)](https://platform.openai.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

An enterprise-grade laboratory and automated evaluation harness designed to systematically measure, benchmark, and optimize Retrieval-Augmented Generation (RAG) architectures.

Built with **LangGraph**, **ChromaDB**, **LangChain**, and **DeepEval**, this repository provides a full end-to-end question-answering assistant along with an isolated **3-Tier Evaluation Harness** targeting both individual pipeline stages and holistic system performance.

---

## 📌 Executive Summary & Motivation

In production RAG systems, diagnosis is impossible without decoupled measurement. When an assistant hallucinates or produces poor responses, engineers face a classic ambiguity:

> **Did the retrieval fail (missing or noisy context), or did the generator fail (poor prompt adherence, hallucination)?**

$$\text{Generation Quality} \le \text{Retrieval Quality}$$

The **RAG Engineering Lab** resolves this through modular, isolated benchmarking across the entire lifecycle:

```
                           ┌───────────────────────────┐
                           │      User Input Query     │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                     =========================================
                     STAGE 1: RETRIEVAL LAYER (Unit Evaluated)
                     Metrics: Contextual Relevancy, Precision, Recall
                     =========================================
                                         │
                                         ▼
                     =========================================
                     STAGE 2: GENERATION LAYER (Unit Evaluated)
                     Metrics: Answer Relevancy, Faithfulness
                     =========================================
                                         │
                                         ▼
                     =========================================
                     STAGE 3: END-TO-END SYSTEM (Integrated)
                     All 5 Metrics Evaluated Across Full Graph
                     =========================================
```

---

## 🏗️ System Architecture

### LangGraph State Workflow

The system is modeled as a compiled state graph (`StateGraph`) with immutable state passing:

```
                  ┌──────────────┐
                  │    START     │
                  └──────┬───────┘
                         │
                         ▼
               ┌───────────────────┐
               │   retrieve_node   │  <-- Vector similarity search in ChromaDB (k=3)
               └─────────┬─────────┘
                         │
                         ▼
               ┌───────────────────┐
               │   generate_node   │  <-- Zero-temperature grounded synthesis (GPT-4.1-mini)
               └─────────┬─────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │     END      │
                  └──────────────┘
```

#### Shared Graph State Schema (`RAGState`)
```python
class RAGState(TypedDict):
    question: str              # User natural language input
    documents: list[Document]  # Retrieved chunk contexts from ChromaDB
    answer: str                # Final synthesized response
```

---

## 📊 The 5 Core RAG Evaluation Metrics

This lab benchmarks systems against the 5 foundational RAG metrics using [DeepEval](https://github.com/confident-ai/deepeval) and LLM-as-a-judge methodologies against verified golden datasets:

```
                     Input Question: "What is corrective RAG?"
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
   ┌───────────────────┐                                 ┌───────────────────┐
   │ Retrieved Context │                                 │ Ground Truth (Ref)│
   └─────────┬─────────┘                                 └─────────┬─────────┘
             │                                                     │
             ├──────────── [3] Contextual Recall ──────────────────┘
             │             (Are all required golden facts retrieved?)
             │
             ├──────────── [1] Contextual Relevancy
             │             (What fraction of context sentences directly help answer Q?)
             │
             ├──────────── [2] Contextual Precision
             │             (Are the best chunks ranked at top positions #1, #2?)
             │
             ▼
   ┌───────────────────┐
   │ Generated Answer  │
   └─────────┬─────────┘
             │
             ├──────────── [4] Answer Relevancy
             │             (Does the answer directly address the user's question?)
             │
             └──────────── [5] Faithfulness
                           (Is every claim strictly supported by retrieved context?)
```

### 1. Contextual Relevancy (Noise vs. Signal Ratio)
* **Definition**: Measures the proportion of sentences within the retrieved context that directly contribute to answering the query.
* **Formula**:
  $$\text{Contextual Relevancy} = \frac{|\text{Relevant Sentences in Context}|}{|\text{Total Sentences in Context}|}$$
* **Inner Working**: An evaluation model parses all retrieved chunks into individual sentences and classifies whether each sentence offers direct value in answering the question.
* **Why It Matters**: Irrelevant sentences consume prompt token budget and distract the LLM's attention heads, increasing latency and cost.
* **Target Threshold**: $\ge 0.70$

---

### 2. Contextual Precision (Rank Quality & Lost-in-the-Middle)
* **Definition**: Measures whether the most relevant chunks are placed at the top of the retrieval list rather than buried at the bottom.
* **Formula** (Weighted Mean Average Precision @ $k$):
  $$\text{Contextual Precision} = \frac{\sum_{k=1}^{K} (\text{Precision}@k \times v_k)}{\text{Total Relevant Nodes}}$$
  where $v_k \in \{0, 1\}$ indicates whether chunk $k$ is relevant.
* **Inner Working**: Ranks chunks from $1$ to $K$, checking alignment against the target answer and penalizing relevant information placed far down the prompt.
* **Why It Matters**: LLMs suffer from the **"Lost in the Middle"** phenomenon ([Liu et al.](https://arxiv.org/abs/2307.03172)) where middle chunks receive lower attention weights than chunks at the extremes.
* **Target Threshold**: $\ge 0.70$

---

### 3. Contextual Recall (Factual Completeness)
* **Definition**: Measures whether the retrieved context contains all essential facts present in the golden expected answer.
* **Formula**:
  $$\text{Contextual Recall} = \frac{|\text{Ground Truth Statements Attributable to Context}|}{|\text{Total Statements in Ground Truth Answer}|}$$
* **Inner Working**: Decomposes the reference answer into atomic factual assertions and verifies if each can be logically deduced strictly from the context.
* **Why It Matters**: If retrieval recall fails, the generator either refuses or hallucinates using pre-trained parametric weights.
* **Target Threshold**: $\ge 0.70$

---

### 4. Answer Relevancy (Question Directness)
* **Definition**: Measures whether the synthesized answer directly and concisely satisfies the user's question without drifting or adding fluff.
* **Formula**:
  $$\text{Answer Relevancy} = \frac{|\text{Statements Relevant to Prompt}|}{|\text{Total Statements in Output}|}$$
* **Inner Working**: Extracts statements from `actual_output` and checks whether they directly resolve the user's question.
* **Why It Matters**: Users demand concise, accurate responses without verbose preambles, disclaimers, or off-topic tangents.
* **Target Threshold**: $\ge 0.70$

---

### 5. Faithfulness (Hallucination Detection)
* **Definition**: Measures whether every factual claim in the generated answer can be strictly grounded in the retrieved context.
* **Formula**:
  $$\text{Faithfulness} = \frac{|\text{Truthful Statements Supported by Context}|}{|\text{Total Statements in Output}|}$$
* **Inner Working**: Extracts all factual assertions in the model's answer and verifies each against the provided context chunks. Any unsupported claim is flagged as a hallucination.
* **Why It Matters**: Hallucinations undermine enterprise credibility. High faithfulness guarantees verifiable provenance.
* **Target Threshold**: $\ge 0.70$

---

## 🧪 3-Tier Evaluation Framework

The repository organizes evaluation into three distinct test suites:

| Suite | File | Scope | Metrics Evaluated |
| :--- | :--- | :--- | :--- |
| **Tier 1: Retriever** | [`evals/test_retrieval_baseline.py`](file:///d:/Agentic%20AI%20Projects/Ezypro%20AI%20Agents%20Internship%20Capstone%20Projects/rag-engineering-lab/evals/test_retrieval_baseline.py) | Retrieval in isolation | Contextual Relevancy, Contextual Precision, Contextual Recall |
| **Tier 2: Generator** | [`evals/test_generator_baseline.py`](file:///d:/Agentic%20AI%20Projects/Ezypro%20AI%20Agents%20Internship%20Capstone%20Projects/rag-engineering-lab/evals/test_generator_baseline.py) | Generation in isolation | Answer Relevancy, Faithfulness |
| **Tier 3: End-to-End** | [`evals/test_rag_baseline.py`](file:///d:/Agentic%20AI%20Projects/Ezypro%20AI%20Agents%20Internship%20Capstone%20Projects/rag-engineering-lab/evals/test_rag_baseline.py) | Full compiled graph | All 5 Core Metrics |

### Baseline Benchmark Results

#### Tier 2: Generator Baseline (`test_generator_baseline.py`)
```text
Aggregate Metrics:
  Metric             Average Score    Pass Rate (@ 0.70)
  ──────────────────────────────────────────────────────
  Answer Relevancy   1.00             100.00% (8/8 passed)
  Faithfulness       1.00             100.00% (8/8 passed)
```
*Conclusion*: The zero-temperature system prompt and strict grounding rules in [`src/graph.py`](file:///d:/Agentic%20AI%20Projects/Ezypro%20AI%20Agents%20Internship%20Capstone%20Projects/rag-engineering-lab/src/graph.py) successfully eliminate hallucinations and ensure 100% directness.

#### Tier 1: Retriever Baseline (`test_retrieval_baseline.py`)
```text
Aggregate Metrics:
  Metric                 Average Score    Pass Rate (@ 0.70)
  ──────────────────────────────────────────────────────────
  Contextual Precision   0.88             87.50% (7/8 passed)
  Contextual Recall      0.81             75.00% (6/8 passed)
  Contextual Relevancy   0.39              0.00% (0/8 passed)
```

#### Tier 3: End-to-End System Baseline (`test_rag_baseline.py`)
```text
Aggregate Metrics:
  Metric                 Average Score    Pass Rate (@ 0.70)
  ──────────────────────────────────────────────────────────
  Answer Relevancy       1.00            100.00% (8/8 passed)
  Faithfulness           1.00            100.00% (8/8 passed)
  Contextual Precision   0.88             87.50% (7/8 passed)
  Contextual Recall      0.88             87.50% (7/8 passed)
  Contextual Relevancy   0.37              0.00% (0/8 passed)
```
*Key Finding*: The generator performs with 100% faithfulness and answer relevancy, but overall test case pass rate is bottlenecked by the retriever introducing noise (relevancy = 0.37) and failing to locate exact alphanumeric needle-in-a-haystack strings (e.g. `ORBIT-LANTERN-42`). Optimization should focus on hybrid search (BM25 + Dense) and cross-encoder re-ranking.

---

## 📂 Repository Layout

```
rag-engineering-lab/
├── .env.example                     # Environment variable template (API keys)
├── .gitignore                       # Clean Git rules (ignores chroma_db & .deepeval)
├── LICENSE                          # Apache 2.0 open-source license
├── README.md                        # Project documentation & benchmarks
├── data/
│   └── books/                       # Domain knowledge base (PDF documents)
│       ├── databases.pdf
│       ├── machine_learning.pdf
│       ├── python.pdf
│       └── rag.pdf
├── evals/
│   ├── dataset.py                   # Golden benchmark test dataset (8 test cases)
│   ├── test_retrieval_baseline.py   # Tier 1: Isolated Retriever evaluation
│   ├── test_generator_baseline.py   # Tier 2: Isolated Generator evaluation
│   └── test_rag_baseline.py         # Tier 3: Full End-to-End system evaluation
└── src/
    ├── chunking.py                  # Text splitter with deterministic chunk_id metadata
    ├── graph.py                     # LangGraph StateGraph pipeline (retrieve -> generate)
    ├── ingestion.py                 # Multi-PDF ingestion with book name tagging
    ├── main.py                      # Interactive CLI terminal assistant
    ├── retriever.py                 # Similarity search retrieval interface
    └── vectorstore.py               # ChromaDB persistence & OpenAI embeddings
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/zainexperience2005/rag-engineering-lab.git
cd rag-engineering-lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# or: .venv\Scripts\activate    # Windows
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Set your OpenAI key in `.env`:
```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
# Optional: specify LLM model (defaults to gpt-4.1-mini)
OPENAI_MODEL=gpt-4.1-mini
```

### 3. Ingest Documents into ChromaDB

```bash
python -m src.vectorstore
```

### 4. Run the Interactive Assistant CLI

Chat with the RAG assistant and inspect live source citations:

```bash
python -m src.main
```

*Interactive Example:*
```text
============================================================
🤖  RAG Engineering Lab - Interactive Assistant
============================================================
Ask any question based on the indexed document library.
Type 'exit' or 'quit' to terminate the session.

You: What does atomicity mean?

Assistant:
Atomicity means that a transaction's changes succeed as a unit or are rolled back. In other words, all operations within a transaction must complete together, or none are applied, ensuring the system remains consistent.

Sources:
  [1] databases.pdf (Page 1) - ID: databases.pdf-page-1-chunk-0
  [2] databases.pdf (Page 1) - ID: databases.pdf-page-1-chunk-1
  [3] databases.pdf (Page 2) - ID: databases.pdf-page-2-chunk-2

You: exit
Exiting session. Goodbye!
```

---

### 5. Running the Evaluation Suites

#### Run Tier 1 (Retriever Only)
```bash
python -m evals.test_retrieval_baseline
```

#### Run Tier 2 (Generator Only)
```bash
python -m evals.test_generator_baseline
```

#### Run Tier 3 (End-to-End System)
```bash
python -m evals.test_rag_baseline
```

---

## 🛠️ Optimization Roadmap

- [ ] **Hybrid Search (BM25 + Dense Vectors)**: Fuse sparse keyword matching with semantic dense search to resolve entity retrieval failures (e.g. `ORBIT-LANTERN-42`).
- [ ] **Cross-Encoder Re-Ranking**: Implement Cohere Rerank or BGE-Reranker to elevate top chunks and filter noise.
- [ ] **Contextual Chunking**: Prepend section titles and document context to chunk embeddings.
- [ ] **Adaptive Query Routing / CRAG**: Trigger web search fallbacks or query reformulations when retrieval confidence is below threshold.

---

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE).