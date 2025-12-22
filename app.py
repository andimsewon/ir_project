"""
Search engine demo UI.
"""
import streamlit as st
import time
import re
import os
import importlib.util

st.set_page_config(
    page_title="Search Engine",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    :root {
        --bg: #fff5f7;
        --bg-accent: #ffeef2;
        --ink: #2b1f24;
        --muted: #6b6b6b;
        --accent: #d96b8c;
        --accent-2: #f3a6bf;
        --card: #ffffff;
        --ring: rgba(217, 107, 140, 0.28);
        --shadow: 0 10px 30px rgba(85, 45, 58, 0.08);
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, rgba(243, 166, 191, 0.25), transparent 60%),
            radial-gradient(1000px 600px at 90% 0%, rgba(217, 107, 140, 0.18), transparent 60%),
            linear-gradient(180deg, var(--bg) 0%, var(--bg-accent) 100%);
        color: var(--ink);
        font-family: "Garamond", "Palatino Linotype", "Book Antiqua", serif;
    }

    .stTextInput > div > div > input {
        border-radius: 999px;
        border: 1px solid rgba(31, 31, 31, 0.15);
        padding: 14px 22px;
        font-size: 17px;
        background: #fff9fb;
        box-shadow: 0 6px 16px rgba(85, 45, 58, 0.08);
        transition: box-shadow 0.3s, border-color 0.3s;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 6px var(--ring);
        border-color: var(--accent);
        outline: none;
    }

    .result-card {
        padding: 18px 20px;
        border: 1px solid rgba(31, 31, 31, 0.08);
        border-radius: 16px;
        background: var(--card);
        box-shadow: var(--shadow);
        margin-bottom: 16px;
        animation: fadeUp 0.4s ease both;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 40px rgba(31, 31, 31, 0.12);
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-title {
        margin: 0;
        font-size: 20px;
        color: #4b2a35;
    }

    .result-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        margin: 8px 0 6px 0;
        font-size: 13px;
        color: #7b5a66;
    }

    .rank-badge {
        background: var(--accent);
        color: #fff;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 12px;
        letter-spacing: 0.3px;
    }

    .doc-id {
        font-family: "Courier New", monospace;
        color: #6b3a4c;
        background: rgba(217, 107, 140, 0.12);
        padding: 2px 8px;
        border-radius: 8px;
    }

    .score-pill {
        background: rgba(243, 166, 191, 0.25);
        color: #7a3b53;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }

    .score-bar {
        width: 100%;
        height: 8px;
        background: rgba(217, 107, 140, 0.18);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 8px;
    }
    .score-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #d96b8c, #f3a6bf);
        border-radius: 999px;
        transition: width 0.3s ease;
    }

    .snippet {
        color: #3b2a30;
        font-size: 14px;
        line-height: 1.6;
        margin-top: 4px;
    }

    .highlight {
        background-color: #ffe1ea;
        padding: 2px 0;
        font-weight: 600;
    }

    .stats-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 8px 0 18px 0;
    }
    .stat-chip {
        background: rgba(217, 107, 140, 0.12);
        color: #7b5a66;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 13px;
    }

    .pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 30px 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    """Load search engine components with caching."""
    from src.indexer import InvertedIndex
    from src.ranker import BM25Ranker
    from src.tfidf_ranker import TFIDFRanker
    from src.reranker import CrossEncoderReranker
    from src.query_expander import QueryExpander
    from src.searcher import SearchEngine
    from src.splade_retriever import SpladeRetriever
    from src.dense_retriever import DenseRetriever

    index_path = "data/index.pkl"
    if not os.path.exists(index_path):
        return None

    index = InvertedIndex()
    index.load(index_path)

    bm25_ranker = BM25Ranker(index)
    tfidf_ranker = TFIDFRanker(index)

    reranker = None
    try:
        reranker = CrossEncoderReranker(model_size="balanced")
    except Exception as exc:
        print(f"[Warning] Reranker disabled: {exc}")

    query_expander = QueryExpander(index, use_embedding=False)

    device = "dml" if importlib.util.find_spec("torch_directml") is not None else None

    splade_retriever = None
    splade_path = "data/splade_index.pt"
    if os.path.exists(splade_path):
        splade_retriever = SpladeRetriever(device=device)
        splade_retriever.load(splade_path)

    dense_retriever = None
    dense_path = "data/dense_index.pt"
    if os.path.exists(dense_path):
        dense_retriever = DenseRetriever(device=device)
        dense_retriever.load(dense_path)

    return SearchEngine(
        index,
        bm25_ranker,
        reranker,
        tfidf_ranker,
        query_expander,
        dense_retriever=dense_retriever,
        splade_retriever=splade_retriever,
    )


def highlight_text(text, query, max_length=300):
    """Highlight query terms in a short snippet."""
    if not text:
        return ""

    query_terms = set(re.findall(r"\b\w+\b", query.lower()))

    if not query_terms:
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    sentences = re.split(r"[.!?]\s+", text)

    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        words = re.findall(r"\b\w+\b", sentence.lower())
        score = sum(1 for w in words if w in query_terms)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if not best_sentence:
        best_sentence = text[:200]

    words_pattern = re.compile(r"\b\w+\b", re.IGNORECASE)

    def highlight_word(match):
        word = match.group(0)
        if word.lower() in query_terms:
            return f'<span class="highlight">{word}</span>'
        return word

    highlighted = words_pattern.sub(highlight_word, best_sentence)

    plain_text = re.sub(r"<[^>]+>", "", highlighted)
    if len(plain_text) > max_length:
        truncated = ""
        tag_open = False
        for char in highlighted:
            if char == "<":
                tag_open = True
            if not tag_open:
                truncated += char
                if len(re.sub(r"<[^>]+>", "", truncated)) >= max_length:
                    break
            if char == ">":
                tag_open = False
        highlighted = truncated + "..."

    return highlighted


def extract_title(doc_text, query):
    """Create a short title from the document text."""
    if not doc_text:
        return "Untitled Document"

    first_sentence = doc_text.split(".")[0].strip()

    query_terms = set(re.findall(r"\b\w+\b", query.lower()))
    words = doc_text.split()

    for i, word in enumerate(words[:50]):
        if word.lower().strip(".,!?;:\"'") in query_terms:
            start = max(0, i - 5)
            end = min(len(words), i + 15)
            title = " ".join(words[start:end])
            if len(title) > 100:
                title = title[:100] + "..."
            return title

    if len(first_sentence) > 100:
        first_sentence = first_sentence[:100] + "..."
    return first_sentence or "Document"


def _available_methods(engine):
    options = []

    if engine and engine.dense_retriever:
        options.append(("Dense", "dense"))
    if engine and engine.splade_retriever:
        options.append(("SPLADE", "splade"))

    options.extend(
        [
            ("BM25", "bm25"),
            ("TF-IDF", "tfidf"),
            ("Hybrid (BM25 + TF-IDF)", "hybrid"),
        ]
    )

    if engine and engine.dense_retriever:
        options.append(("Hybrid (BM25 + Dense)", "hybrid_dense"))
    if engine and engine.splade_retriever:
        options.append(("Hybrid (BM25 + SPLADE)", "hybrid_splade"))

    return options


def _method_descriptions():
    return {
        "dense": "Dense retrieval with ANN (FAISS when available).",
        "splade": "Sparse neural retrieval with SPLADE.",
        "bm25": "Classic lexical BM25 ranking.",
        "tfidf": "TF-IDF lexical ranking.",
        "hybrid": "Linear fusion of BM25 + TF-IDF.",
        "hybrid_dense": "Linear fusion of BM25 + Dense.",
        "hybrid_splade": "Linear fusion of BM25 + SPLADE.",
    }


def main():
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "results_per_page" not in st.session_state:
        st.session_state.results_per_page = 10
    if "method_choice" not in st.session_state:
        st.session_state.method_choice = "dense"
    if "use_reranker_opt" not in st.session_state:
        st.session_state.use_reranker_opt = False
    if "use_expansion_opt" not in st.session_state:
        st.session_state.use_expansion_opt = False
    if "hybrid_weight" not in st.session_state:
        st.session_state.hybrid_weight = 0.6
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if st.session_state.pending_query:
        st.session_state.search_input = st.session_state.pending_query
        st.session_state.pending_query = None
        st.session_state.search_results = None
        st.session_state.current_page = 1

    index_path = "data/index.pkl"
    if not os.path.exists(index_path):
        st.error("Required index files are missing.")
        st.code("python download_data.py\npython build_index.py", language="bash")
        return

    try:
        engine = load_engine()
    except Exception as exc:
        st.error(f"Engine load failed: {exc}")
        st.code("python download_data.py\npython build_index.py", language="bash")
        return

    if engine is None:
        st.error("Index not found. Run the setup commands first.")
        st.code("python download_data.py\npython build_index.py", language="bash")
        return

    method_options = _available_methods(engine)
    if not method_options:
        st.error("No retrieval methods are available.")
        return

    method_keys = [key for _, key in method_options]
    if st.session_state.method_choice not in method_keys:
        st.session_state.method_choice = method_keys[0]

    reranker_available = engine.reranker is not None
    if not reranker_available:
        st.session_state.use_reranker_opt = False

    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(
                "<div style=\"text-align: center; margin-bottom: 30px;\">"
                "<div style=\"font-size: 90px; font-weight: 800; font-style: italic; color: #ff5c8a; letter-spacing: 2px;\">SAP</div>"
                "<div style=\"font-size: 20px; color: #ff8fb0; margin-top: 6px;\">Search Anything Positively</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            with st.form("search_form", clear_on_submit=False):
                query_input = st.text_input(
                    "",
                    value=st.session_state.get("search_input", ""),
                    placeholder="Enter your query...",
                    key="search_input",
                    label_visibility="collapsed",
                )

                method_labels = [label for label, _ in method_options]
                current_index = method_keys.index(st.session_state.method_choice)
                selected_label = st.selectbox(
                    "Search method",
                    method_labels,
                    index=current_index,
                )
                selected_key = dict(method_options)[selected_label]
                st.session_state.method_choice = selected_key

                st.markdown("<br>", unsafe_allow_html=True)
                toggle_cols = st.columns(2)
                with toggle_cols[0]:
                    st.toggle(
                        "Reranker",
                        value=st.session_state.use_reranker_opt,
                        key="use_reranker_opt",
                        disabled=not reranker_available,
                    )
                with toggle_cols[1]:
                    st.toggle(
                        "Query Expansion",
                        value=st.session_state.use_expansion_opt,
                        key="use_expansion_opt",
                    )

                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                with col_btn2:
                    search_clicked = st.form_submit_button("Search", use_container_width=True, type="primary")

            query = query_input.strip() if query_input else ""

            chips = [
                f"Method: {selected_label}",
                f"Reranker: {'ON' if st.session_state.use_reranker_opt else 'OFF'}",
                f"Query Expansion: {'ON' if st.session_state.use_expansion_opt else 'OFF'}",
            ]
            st.caption(" | ".join(chips))
            st.caption("Tip: press Search to run the query.")

            if st.session_state.search_results:
                if st.button("Clear results", use_container_width=True):
                    st.session_state.search_results = None
                    st.session_state.current_page = 1
                    st.rerun()

    with st.sidebar:
        st.header("Advanced Settings")

        if engine:
            st.markdown("**Index Stats**")
            st.markdown(f"- Documents: {engine.index.total_docs:,}")
            st.markdown(f"- Vocabulary: {len(engine.index.posting_list):,}")
            st.markdown(f"- Avg doc length: {engine.index.avg_doc_len:.1f}")

        st.markdown("---")
        st.markdown("**Retrieval Options**")
        st.caption("Choose a method in the main panel.")
        method_desc = _method_descriptions().get(st.session_state.method_choice, "")
        if method_desc:
            st.caption(f"Current: {method_desc}")

        if not reranker_available:
            st.caption("Reranker unavailable")

        if st.session_state.method_choice in {"hybrid", "hybrid_dense", "hybrid_splade"}:
            st.session_state.hybrid_weight = st.slider(
                "Hybrid weight (BM25)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.hybrid_weight),
                step=0.05,
            )

    if search_clicked and query.strip():
        status = st.status("Searching...", expanded=False) if hasattr(st, "status") else None
        with st.spinner("Searching..."):
            start_time = time.time()

            use_reranker = st.session_state.use_reranker_opt
            use_expansion = st.session_state.use_expansion_opt
            method = st.session_state.method_choice

            try:
                result = engine.search(
                    query,
                    top_k=100,
                    method=method,
                    use_reranker=use_reranker,
                    use_query_expansion=use_expansion,
                    hybrid_weight=st.session_state.hybrid_weight,
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                return

            elapsed = time.time() - start_time
            st.session_state.search_results = result
            st.session_state.search_time = elapsed
            st.session_state.current_page = 1
            if status:
                status.update(label=f"Search complete ({elapsed:.2f}s)", state="complete")

    if st.session_state.search_results:
        result = st.session_state.search_results

        st.markdown(
            f"""
        <div class="stats-bar">
            <span class="stat-chip">Results: {len(result['results']):,}</span>
            <span class="stat-chip">Time: {st.session_state.search_time:.3f}s</span>
            <span class="stat-chip">Method: {result['method']}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if result.get("expanded_query") and result["expanded_query"] != result["query"]:
            st.info(f"Expanded query: **{result['expanded_query']}**")

        if not result["results"]:
            st.warning("No results found. Try another query.")
            st.markdown("**Suggested queries:**")
            examples = ["machine learning", "artificial intelligence", "world war", "climate change"]
            cols = st.columns(len(examples))
            for i, ex in enumerate(examples):
                if cols[i].button(ex, key=f"ex_no_results_{i}"):
                    st.session_state.pending_query = ex
                    st.rerun()
        else:
            total_results = len(result["results"])
            total_pages = (total_results - 1) // st.session_state.results_per_page + 1
            start_idx = (st.session_state.current_page - 1) * st.session_state.results_per_page
            end_idx = start_idx + st.session_state.results_per_page
            page_results = result["results"][start_idx:end_idx]
            max_score = max((r["score"] for r in result["results"]), default=1.0) or 1.0
            method_label = result.get("method", "Score")

            for r in page_results:
                doc_id = r["doc_id"]
                score = r["score"]
                snippet = r["snippet"]
                full_text = engine.get_document(doc_id)
                score_pct = min(100.0, (score / max_score) * 100.0) if max_score > 0 else 0.0

                title = extract_title(full_text, result["query"])
                highlighted_title = highlight_text(title, result["query"], max_length=150)

                highlighted_snippet = highlight_text(snippet, result["query"])

                st.markdown(
                    f"""
                <div class="result-card">
                    <h3 class="result-title">{highlighted_title}</h3>
                    <div class="result-meta">
                        <span class="rank-badge">Rank {r['rank']}</span>
                        <span class="doc-id">{doc_id}</span>
                        <span class="score-pill">Score ({method_label}) {score:.4f}</span>
                    </div>
                    <div class="snippet">
                        {highlighted_snippet if highlighted_snippet else snippet}
                    </div>
                    <div class="score-bar" title="Relative score">
                        <div class="score-bar-fill" style="width: {score_pct:.1f}%;"></div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                with st.expander("Full document", expanded=False):
                    st.text_area("", full_text[:5000], height=200, key=f"full_{doc_id}", disabled=True)

            if total_pages > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                pagination_cols = st.columns([1, 2, 1])

                with pagination_cols[0]:
                    if st.button("Prev", disabled=(st.session_state.current_page == 1)):
                        st.session_state.current_page -= 1
                        st.rerun()

                with pagination_cols[1]:
                    st.markdown(
                        f"<div style=\"text-align: center; padding: 10px;\">"
                        f"Page {st.session_state.current_page} / {total_pages}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with pagination_cols[2]:
                    if st.button("Next", disabled=(st.session_state.current_page >= total_pages)):
                        st.session_state.current_page += 1
                        st.rerun()

            st.markdown("---")
            new_per_page = st.selectbox(
                "Results per page",
                [5, 10, 20, 30, 50],
                index=[5, 10, 20, 30, 50].index(st.session_state.results_per_page),
                key="per_page_selector",
            )
            if new_per_page != st.session_state.results_per_page:
                st.session_state.results_per_page = new_per_page
                st.session_state.current_page = 1
                st.rerun()


if __name__ == "__main__":
    main()
