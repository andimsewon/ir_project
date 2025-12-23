# SAP (Search Anything Positively)

SAP is a full-stack Information Retrieval (IR) research prototype implemented from scratch. It provides a controlled, reproducible environment to study classical ranking (BM25/TF-IDF), sparse neural retrieval (SPLADE), dense retrieval with ANN (FAISS), and Cross-Encoder reranking, with both quantitative evaluation and qualitative inspection via Streamlit/CLI.

---

## Table of Contents

- Project Overview
- Research Framing
- System Architecture
- Key Features
- Getting Started (Windows / macOS)
- Evaluation (TREC)
- Project Structure
- Constraints
- Troubleshooting

---

## Project Overview

### Goals
- Build a complete IR pipeline end-to-end: indexing, retrieval, ranking, reranking, evaluation, and UI
- Enable apples-to-apples comparisons across classical, sparse neural, and dense retrieval methods
- Support reproducible experiments on a standard benchmark dataset

### Highlights
- From-scratch indexing and ranking logic (no external search engines)
- Modular retrievers and rerankers for controlled experiments
- Standard IR metrics via `pytrec_eval`
- Streamlit UI for qualitative analysis; CLI for fast iteration

### Dataset
- `wikir/en1k` from `ir_datasets`
- Documents: 369,712
- Queries: training 1,444 / validation 100 / test 100
- Relevance: 0 (not relevant), 1 (relevant), 2 (highly relevant)

---

## Research Framing

### Research Questions
- How do classical lexical baselines compare to sparse neural and dense retrieval on `wikir/en1k`?
- What is the effect of a Cross-Encoder reranker on top-k results?
- How does ANN indexing impact evaluation quality and runtime?

### Methodology
- Build consistent indices per retrieval family (BM25/TF-IDF, SPLADE, Dense+ANN)
- Evaluate with identical train/validation/test splits and metrics
- Record outputs in `results/` for method-level comparisons

### Reproducibility Notes
- Dataset and splits come directly from `ir_datasets`
- Index artifacts are stored under `data/`
- All evaluation is scripted via `run_eval.py`

---

## System Architecture

### End-to-End Pipeline

User query input
-> Query Processor (tokenization, optional query expansion)
-> Retrieval/Ranking (BM25/TF-IDF/Dense/SPLADE)
-> Reranker (optional, Cross-Encoder)
-> Output (snippets/highlights, pagination)

### Core Components
- Inverted Index: `src/indexer.py`
- Rankers: `src/ranker.py` (BM25), `src/tfidf_ranker.py`
- Search Engine: `src/searcher.py`
- Query Expander: `src/query_expander.py`
- Reranker: `src/reranker.py`
- Dense/SPLADE: `src/dense_retriever.py`, `src/splade_retriever.py`
- ANN (FAISS): `src/dense_retriever.py`
- UI: `app.py` (Streamlit), `cli_search.py` (CLI)

---

## Key Features

### Required Features
- Inverted index build (posting list, DF, doc length, doc store)
- BM25 ranking (k1=1.5, b=0.75)
- TF-IDF ranking
- Streamlit web UI (method selection, reranker/query expansion toggles, score type display)

### Additional Features
- Query expansion (synonym/co-occurrence/embedding-based)
- Cross-Encoder reranking (`BAAI/bge-reranker-base` by default)
- Highlighting, pagination, option toggles
- Score explainability: term-level contributions for BM25/TF-IDF in the UI
- Sidebar Query Analyzer: token/DF/IDF table for the current query
- Related Queries mined from top results (TF×IDF), as one-click buttons
- Download current results in TREC run format
- Dense retrieval + ANN (optional, `BAAI/bge-base-en-v1.5`, FAISS HNSW)
- SPLADE (required, `naver/splade-cocondenser-ensembledistil`)

---

## Getting Started (Windows / macOS)

### 1) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

For ANN (FAISS):

```bash
pip install faiss-cpu
```

### 3) Download data

```bash
python download_data.py
```

Alternative (manual download):

```bash
python download_data_direct.py
```

### 4) Build the BM25 index

```bash
python build_index.py
```

### 4-1) Build the SPLADE index (required, GPU recommended)

Using Intel Arc A770 (DirectML):

```bash
pip install torch-directml
python build_splade_index.py --device dml
```

CPU:

```bash
python build_splade_index.py --device cpu
```

You must build the SPLADE index for search to work.
Streamlit/CLI will automatically use DirectML if `torch-directml` is installed.

### 4-2) Build Dense + ANN index (optional)

If FAISS is installed, an ANN index is generated automatically and used during evaluation.
ANN search runs on CPU, and embedding encoding uses Intel Arc GPU with `--device dml`.
Currently Streamlit/CLI run SPLADE only; Dense ANN is for experiments.

```bash
python build_dense_index.py --device dml
```

If you hit GPU memory errors, reduce batch size/length:

```bash
python build_dense_index.py --device dml --batch-size 8 --max-length 256
```

### 5) Run the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 5-quick) Lightweight BM25-only UI

A minimal Streamlit app that uses only BM25 and avoids importing heavy dependencies (transformers/torch). As long as the BM25 index exists, you can run it immediately:

```bash
pip install -r requirements-bm25.txt   # install minimal, lightweight deps
python build_index.py                   # first time only, if needed
streamlit run app_bm25.py
```

This mode fixes the method to BM25 and disables reranker/query expansion.

### 5-1) Web UI usage (method-selectable, default Dense)

1) Select a retrieval method (Dense is default)
2) Enter a query and press Enter or click `Search`
3) Check title/snippet and the score type label in result cards
4) Move pages with pagination
5) Toggle Reranker / Query Expansion options
6) For hybrid methods, adjust the BM25 weight in the sidebar
7) Use `Clear results` to reset the view

### 5-2) Evaluation Dashboard (research comparison)

Run a Streamlit page to evaluate multiple methods at once (MAP/P@k/R@k/nDCG), display tables and bar charts, and download TREC runs per method:

```bash
streamlit run app_eval.py
```

### 6) Run the CLI (optional)

```bash
python cli_search.py --top-k 10
```

---

## One-line macOS Run

For a quick start on macOS that sets up a venv, installs deps with fallbacks, builds the index if needed, and launches the web UI:

```bash
bash scripts/run_mac.sh
```

If dependency installation fails due to native builds (e.g., pytrec_eval or faiss-cpu), the script falls back to a minimal set required for the web UI. For evaluation, install the extras later:

```bash
source .venv/bin/activate
pip install pytrec_eval faiss-cpu
```

---

## Mac Demo Quick Setup (Recommended)

### Option A: Copy data/index built on Windows

1) Copy these files from Windows:
   - `data/documents.tsv`
   - `data/queries_*.tsv`
   - `data/qrels_*.tsv`
   - `data/index.pkl`
   - (optional) `data/dense_index.pt`, `data/splade_index.pt`
2) Copy them to the same paths on Mac
3) On Mac: create venv + `pip install -r requirements.txt`
4) Rehearsal:

```bash
python check_data.py
streamlit run app.py
```

### Option B: Download + build on Mac

```bash
python download_data.py
python build_index.py
streamlit run app.py
```

---

## Evaluation (TREC)

### Run

```bash
python run_eval.py --split validation
python run_eval.py --split test
```

SPLADE evaluation:

```bash
python run_eval.py --split validation --method splade
python run_eval.py --split test --method splade
```

Dense ANN evaluation:

```bash
python run_eval.py --split validation --method dense_ann
python run_eval.py --split test --method dense_ann
```

BM25 vs Reranker (default):

```bash
python run_eval.py --split validation
python run_eval.py --split test
```

### Output locations
- `results/summary_validation.txt`
- `results/summary_test.txt`
- `results/bm25_*.txt`, `results/tfidf_*.txt`, `results/hybrid_*.txt`, `results/rerank_*.txt`
- `results/summary_splade_*.txt`, `results/splade_*.txt`
- `results/summary_dense_ann_*.txt`, `results/dense_ann_*.txt`

### Metrics
- MAP, Precision@k, Recall@k, nDCG

### Result Summary (current results)

Validation (BM25 vs BM25+Reranker)

| Metric | BM25 | BM25 + Reranker |
| --- | --- | --- |
| MAP | 0.1495 | 0.1372 |
| P@10 | 0.2080 | 0.1940 |
| nDCG@10 | 0.3078 | 0.3178 |

Test (BM25 vs BM25+Reranker)

| Metric | BM25 | BM25 + Reranker |
| --- | --- | --- |
| MAP | 0.1754 | 0.1535 |
| P@10 | 0.2120 | 0.1990 |
| nDCG@10 | 0.3584 | 0.3405 |

Note: In this run, the reranker improved nDCG@10 on validation but reduced MAP/P@10, and it dropped overall on test.

---

## Project Structure

```
ir_project/
├─ app.py                   # Streamlit UI
├─ cli_search.py            # Terminal UI
├─ download_data.py         # ir_datasets download
├─ download_data_direct.py  # manual download alternative
├─ process_manual_download.py
├─ build_index.py           # BM25 index build
├─ build_dense_index.py     # Dense + ANN index build
├─ build_splade_index.py    # SPLADE index build
├─ run_eval.py              # evaluation
├─ check_data.py            # data sanity check
├─ requirements.txt
├─ src/
│  ├─ indexer.py
│  ├─ tokenizer.py
│  ├─ ranker.py
│  ├─ tfidf_ranker.py
│  ├─ searcher.py
│  ├─ query_expander.py
│  ├─ reranker.py
│  ├─ dense_retriever.py
│  ├─ splade_retriever.py
│  └─ evaluator.py
├─ data/                    # documents.tsv, queries_*.tsv, qrels_*.tsv, index.pkl, splade_index.pt, dense_index.pt, dense_index.faiss
└─ results/                 # evaluation outputs
```

---

## Constraints

- No external search engine libraries (Elasticsearch/Lucene/Solr/Indri)
- Use only HuggingFace pretrained models
- Indexing/ranking logic implemented from scratch

---

## Troubleshooting

- Index not found
  - `python download_data.py` -> `python build_index.py` -> `python build_splade_index.py`
- Slow model download
  - Run once before demos to warm the cache
- Slow on macOS
  - CPU-only is sufficient for demos
- Unstable network
  - Option A (copy data/index) recommended
- faiss-cpu install failure (Windows)
  - If install fails, skip Dense ANN and build with `--no-ann`.
