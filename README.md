# 🔬 RAG Engineering Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3%2B-green.svg)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com/)
[![DeepEval](https://img.shields.io/badge/Evals-DeepEval-purple.svg)](https://confident-ai.com/)
[![OpenAI](https://img.shields.io/badge/Embeddings-text--embedding--3--small-black.svg)](https://platform.openai.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

An engineering-grade testbed and evaluation harness designed to systematically measure, benchmark, and optimize the **Retrieval Layer** in Retrieval-Augmented Generation (RAG) pipelines.

---

## 📌 Executive Summary

Most RAG failures (hallucinations, off-target replies, incomplete answers) originate not in the generator LLM, but in the **retrieval stage**:

$$\text{Generation Quality} \le \text{Retrieval Quality}$$

If the retriever returns irrelevant sentences (noise), misses essential facts (low recall), or ranks the most useful context below distracting chunks (poor precision), the downstream LLM cannot reliably generate accurate answers.

**RAG Engineering Lab** decouples retrieval from generation. It isolates the retrieval layer, benchmarks similarity search against a curated golden dataset, and quantifies performance across three industry-standard metrics: **Contextual Relevancy**, **Contextual Precision**, and **Contextual Recall**.

---

## 🏗️ System Architecture

```
                       ┌────────────────────────────────────────┐
                       │           Document Ingestion           │
                       │   data/books/*.pdf (PyPDFLoader)       │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │          Document Chunking             │
                       │   RecursiveCharacterTextSplitter       │
                       │   size=1000, overlap=200, chunk_id     │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │       Vector Store & Indexing          │
                       │   OpenAI text-embedding-3-small        │
                       │   Persistent ChromaDB (collection)     │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │            Retriever API               │
                       │   VectorStoreRetriever (k=3)           │
                       └───────────────────┬────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
                     ▼                                           ▼
        ┌─────────────────────────┐               ┌──────────────────────────────┐
        │     Runtime Query       │               │      DeepEval Harness        │
        │   Natural Language Q    │               │   evals/dataset.py (Golden)  │
        │   Top-k Context Output  │               │   evals/test_retrieval_baseline  │
        └─────────────────────────┘               └──────────────┬───────────────┘
                                                                 │
                                                                 ▼
                                                  ┌──────────────────────────────┐
                                                  │       Evaluation Metrics     │
                                                  │   • Contextual Relevancy     │
                                                  │   • Contextual Precision     │
                                                  │   • Contextual Recall        │
                                                  └──────────────────────────────┘
```

---

## 📊 Evaluation Metrics Deep-Dive

This laboratory uses [DeepEval](https://github.com/confident-ai/deepeval) to evaluate retrieval context quality using LLM-as-a-judge methodologies against verified ground truth.

```
       User Query: "What is corrective RAG?"
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│       Retrieved Context      │   │     Expected Ground Truth    │
│  [Node 1, Node 2, Node 3]    │   │  "Corrective RAG assesses... │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               │                                  │
               ├─────────────── Contextual Recall ┘
               │                (Are all ground-truth facts present?)
               │
               ├── Contextual Relevancy
               │   (What fraction of context sentences directly help answer Q?)
               │
               └── Contextual Precision
                   (Are the most relevant nodes ranked at the top?)
```

### 1. Contextual Relevancy (Signal-to-Noise Ratio)

* **Definition**: Quantifies the proportion of sentences in the retrieved context that are strictly relevant to answering the user query.
* **Formula**:
  $$\text{Contextual Relevancy} = \frac{|\text{Relevant Sentences in Context}|}{|\text{Total Sentences in Context}|}$$
* **Inner Working**:
  1. The evaluation LLM segments retrieved context chunks into atomic sentences.
  2. Each sentence is evaluated for whether it contains information that directly answers the question.
  3. The final score is the ratio of relevant sentences to total retrieved sentences.
* **Why It Matters**:
  Large context windows do not prevent distractors. Irrelevant sentences dilute LLM attention, waste token budget, and trigger confabulation.
* **Production Threshold**: `0.70`

---

### 2. Contextual Precision (Rank-Aware Retrieval Quality)

* **Definition**: Measures whether the most relevant chunks are ranked at the top of the retrieval list rather than buried at lower positions.
* **Formula** (Weighted Mean Average Precision @ $k$):
  $$\text{Contextual Precision} = \frac{\sum_{k=1}^{K} (\text{Precision}@k \times v_k)}{\text{Total Relevant Nodes}}$$
  where $v_k \in \{0, 1\}$ indicates whether chunk $k$ is relevant, and $\text{Precision}@k = \frac{\text{Relevant Nodes in Top } k}{k}$.
* **Inner Working**:
  1. Evaluates each retrieved node in sequential order ($k = 1, 2, \dots, K$).
  2. Checks whether node $k$ contributes toward the expected ground-truth answer.
  3. Penalizes the score if irrelevant nodes precede relevant ones.
* **Why It Matters**:
  LLMs suffer from the **"Lost in the Middle"** phenomenon ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)). Information positioned at the very beginning of the context prompt receives the highest attention weights. Chunks at Rank #1 must be the most informative.
* **Production Threshold**: `0.70`

---

### 3. Contextual Recall (Information Completeness)

* **Definition**: Measures whether the retrieved context contains all necessary facts required to produce the ground-truth answer.
* **Formula**:
  $$\text{Contextual Recall} = \frac{|\text{Ground Truth Statements Attributable to Context}|}{|\text{Total Statements in Ground Truth Answer}|}$$
* **Inner Working**:
  1. Breaks the golden `expected_answer` down into distinct factual claims.
  2. Evaluates whether each claim can be logically deduced or attributed to the `retrieval_context`.
  3. Outputs the fraction of supported claims.
* **Why It Matters**:
  If retrieval recall is inadequate, the generator LLM either refuses to answer or relies on pre-trained parametric memory, causing hallucinations.
* **Production Threshold**: `0.70`

---

## 📈 Baseline Benchmark Results

Running the baseline retrieval test suite (`k=3`, `chunk_size=1000`, `chunk_overlap=200`, `text-embedding-3-small`) produced the following empirical baseline:

| Metric | Average Score | Pass Rate (@ 0.70) | Status | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Contextual Precision** | **0.88** | **87.50%** (7/8 passed) | 🟢 Strong | When relevant chunks are found, they are consistently ranked at Position #1. |
| **Contextual Recall** | **0.81** | **75.00%** (6/8 passed) | 🟡 Moderate | Core conceptual facts are captured, but exact identifiers are missed. |
| **Contextual Relevancy** | **0.39** | **0.00%** (0/8 passed) | 🔴 Degraded | Chunks contain excessive surrounding text, headers, and meta-text noise. |

### 🔍 Failure Case Analysis (Test Case 8)

* **Question**: *"What is the corpus project code name?"*
* **Expected Answer**: `"ORBIT-LANTERN-42."`
* **Score**: Relevancy = 0.17 | Precision = 0.00 | Recall = 0.00
* **Root Cause**:
  Dense vector embeddings (`text-embedding-3-small`) optimize for semantic similarity rather than exact keyword / alphanumeric matching. The similarity search matched generic corpus book headers rather than the specific synthetic code-name token.
* **Prescribed Solution**: Implement **Hybrid Search (BM25 + Dense Vectors)** and **Cross-Encoder Re-ranking**.

---

## 📂 Repository Layout

```
rag-engineering-lab/
├── .env.example                  # Template for required environment variables
├── .gitignore                    # Git ignore rules (includes chroma_db & .deepeval)
├── LICENSE                       # Apache 2.0 open-source license
├── README.md                     # Comprehensive project documentation
├── data/
│   └── books/                    # Evaluation PDF documents
│       ├── databases.pdf
│       ├── machine_learning.pdf
│       ├── python.pdf
│       └── rag.pdf
├── evals/
│   ├── dataset.py                # Curated golden benchmark dataset (questions & ground truth)
│   └── test_retrieval_baseline.py# DeepEval retrieval evaluation harness
└── src/
    ├── chunking.py               # Document splitter with deterministic chunk_id metadata
    ├── ingestion.py              # PDF document loader preserving book provenance
    ├── retriever.py              # Query interface for top-k vector similarity search
    └── vectorstore.py            # ChromaDB persistent store & OpenAI embeddings pipeline
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/zainexperience2005/rag-engineering-lab.git
cd rag-engineering-lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # On Linux/macOS
# or: .venv\Scripts\activate     # On Windows
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and insert your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Ingest and Index Documents

Run `vectorstore.py` to parse PDFs, split into overlapping chunks, generate embeddings, and persist them in ChromaDB:

```bash
python -m src.vectorstore
```

*Output:*
```text
Loading raw corpus documents...
Loading: databases.pdf
Loading: machine_learning.pdf
Loading: python.pdf
Loading: rag.pdf
Splitting documents into chunks...
Embedding and persisting 54 chunks into ChromaDB at 'chroma_db'...
Successfully stored 54 chunks in ChromaDB.
```

### 4. Test Single Query Retrieval

Verify that similarity search returns relevant context:

```bash
python -m src.retriever
```

### 5. Run the DeepEval Benchmark Suite

Execute the complete retrieval evaluation harness:

```bash
python -m evals.test_retrieval_baseline
```

---

## 🗺️ Engineering Optimization Roadmap

To improve upon the baseline scores (targeting **Relevancy > 0.70**, **Precision > 0.90**, **Recall > 0.90**):

- [ ] **Hybrid Search**: Combine BM25 sparse keyword retrieval with dense vector embeddings using Reciprocal Rank Fusion (RRF).
- [ ] **Cross-Encoder Re-Ranking**: Use a cross-encoder model (e.g., `bge-reranker-large` or Cohere Rerank) to re-order the top-15 retrieved nodes down to top-3.
- [ ] **Contextual Chunking**: Prepend parent document summaries or section titles to each chunk before embedding.
- [ ] **Dynamic Top-$k$ / Score Thresholding**: Prune low-similarity chunks rather than returning a rigid $k=3$.
- [ ] **Query Transformation / CRAG**: Implement Corrective RAG query rewriting to expand acronyms and refine ambiguous queries.

---

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE).