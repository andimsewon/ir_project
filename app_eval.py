"""
Interactive evaluation dashboard for the search engine.
Run with: streamlit run app_eval.py
"""
import os
import time
import streamlit as st

from src.indexer import InvertedIndex
from src.ranker import BM25Ranker
from src.searcher import SearchEngine
from src.evaluator import Evaluator

INDEX_PATH = "data/index.pkl"
SPLADE_INDEX_PATH = "data/splade_index.pt"
DENSE_INDEX_PATH = "data/dense_index.pt"


st.set_page_config(
    page_title="IR Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _load_index():
    idx = InvertedIndex()
    idx.load(INDEX_PATH)
    return idx


@st.cache_resource
def _load_splade(device=None):
    from src.splade_retriever import SpladeRetriever
    retr = SpladeRetriever(device=device)
    retr.load(SPLADE_INDEX_PATH)
    return retr


@st.cache_resource
def _load_dense(device=None, ann_only=False):
    from src.dense_retriever import DenseRetriever
    retr = DenseRetriever(device=device, ann_enabled=ann_only)
    retr.load(DENSE_INDEX_PATH)
    return retr


def _available_methods():
    methods = ["BM25"]
    # Reranker availability handled dynamically later
    if os.path.exists(SPLADE_INDEX_PATH):
        methods.append("SPLADE")
    if os.path.exists(DENSE_INDEX_PATH):
        methods.append("Dense-ANN")
    return methods


st.title("IR Evaluation Dashboard")
st.caption("Compare retrieval methods using MAP, P@k, R@k, nDCG@k")

if not os.path.exists(INDEX_PATH):
    st.error("Index not found. Please build it first: python build_index.py")
    st.stop()

split = st.selectbox("Dataset split", ["training", "validation", "test"], index=1)
k_list = st.multiselect("k values", [5, 10, 20, 50], default=[5, 10, 20])
methods = st.multiselect("Methods", _available_methods(), default=["BM25"]) 

st.markdown("---")
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    top_k = st.number_input("Top K per query", min_value=10, max_value=1000, value=100, step=10)
with colB:
    use_reranker = st.toggle("BM25 + Reranker", value=False)
with colC:
    run_btn = st.button("Run Evaluation", type="primary")


def evaluate_method(method: str, evaluator: Evaluator):
    index = _load_index()
    bm25 = BM25Ranker(index)

    # Build engine per method
    engine = None
    reranker = None
    if method == "BM25":
        if use_reranker:
            try:
                from src.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker(model_size="balanced")
            except Exception:
                reranker = None
        engine = SearchEngine(index, bm25, reranker=reranker)
        run = evaluator.generate_run(engine, use_reranker=bool(reranker), top_k=top_k)
        return run, method if not reranker else "BM25+Reranker"

    if method == "SPLADE":
        try:
            splade = _load_splade()
        except Exception as exc:
            st.error(f"SPLADE not available: {exc}")
            return None, method
        engine = SearchEngine(index, bm25, splade_retriever=splade)
        run = evaluator.generate_run(engine, use_reranker=False, top_k=top_k, method="splade")
        return run, method

    if method == "Dense-ANN":
        try:
            dense = _load_dense(ann_only=True)
            if getattr(dense, "ann_index", None) is None:
                st.error("Dense ANN index is not available. Rebuild with faiss installed.")
                return None, method
        except Exception as exc:
            st.error(f"Dense retriever not available: {exc}")
            return None, method
        engine = SearchEngine(index, bm25, dense_retriever=dense)
        run = evaluator.generate_run(engine, use_reranker=False, top_k=top_k, method="dense")
        return run, method

    st.warning(f"Unknown method: {method}")
    return None, method


qrels_path = f"data/qrels_{split}.tsv"
queries_path = f"data/queries_{split}.tsv"
if not (os.path.exists(qrels_path) and os.path.exists(queries_path)):
    st.error(f"Missing qrels or queries for split '{split}'. Download data first.")
    st.stop()


if run_btn:
    status = st.status("Running evaluation...", expanded=True) if hasattr(st, "status") else None
    start = time.time()
    evaluator = Evaluator(qrels_path, queries_path)

    results_table = []
    per_method_runs = {}

    for m in methods:
        if status:
            status.update(label=f"Evaluating {m}...")
        run, label = evaluate_method(m, evaluator)
        if not run:
            continue
        per_method_runs[label] = run
        scores = evaluator.evaluate(run, k_list=k_list)
        results_table.append({"Method": label, **scores})

    elapsed = time.time() - start
    if status:
        status.update(label=f"Done in {elapsed:.2f}s", state="complete")

    if not results_table:
        st.warning("No results produced.")
        st.stop()

    # Display results table
    st.subheader("Results")
    st.dataframe(results_table, use_container_width=True)

    # Plot a couple of key metrics if present
    try:
        import pandas as pd
        df = pd.DataFrame(results_table).set_index("Method")
        plot_cols = [c for c in df.columns if c.lower().startswith("map") or c.lower().startswith("ndcg")]
        if plot_cols:
            st.bar_chart(df[plot_cols])
    except Exception:
        pass

    # Allow download of runs (TREC format per method)
    st.markdown("---")
    st.subheader("Download TREC Runs")
    for label, run in per_method_runs.items():
        lines = []
        for qid, results in run.items():
            for rank, (doc_id, score) in enumerate(results, 1):
                lines.append(f"{qid} Q0 {doc_id} {rank} {score:.6f} {label}\n")
        st.download_button(
            f"Download {label}",
            data="".join(lines),
            file_name=f"run_{label.replace(' ','_').lower()}_{split}.txt",
            mime="text/plain",
        )

