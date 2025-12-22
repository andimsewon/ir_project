"""
Lightweight Streamlit app that uses only BM25.
No heavy dependencies (transformers/torch/etc.) are imported.
Run with: streamlit run app_bm25.py
"""
import os
import re
import time
import math

import streamlit as st


st.set_page_config(
    page_title="Search Engine (BM25)",
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

    .result-card { padding: 18px 20px; border: 1px solid rgba(31,31,31,0.08); border-radius: 16px; background: var(--card); box-shadow: var(--shadow); margin-bottom: 16px; }
    .result-title { margin: 0; font-size: 20px; color: #4b2a35; }
    .result-meta { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 8px 0 6px 0; font-size: 13px; color: #7b5a66; }
    .rank-badge { background: var(--accent); color: #fff; border-radius: 999px; padding: 2px 10px; font-size: 12px; letter-spacing: 0.3px; }
    .doc-id { font-family: "Courier New", monospace; color: #6b3a4c; background: rgba(217, 107, 140, 0.12); padding: 2px 8px; border-radius: 8px; }
    .score-pill { background: rgba(243, 166, 191, 0.25); color: #7a3b53; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .score-bar { width: 100%; height: 8px; background: rgba(217, 107, 140, 0.18); border-radius: 999px; overflow: hidden; margin-top: 8px; }
    .score-bar-fill { height: 100%; background: linear-gradient(90deg, #d96b8c, #f3a6bf); border-radius: 999px; }
    .snippet { color: #3b2a30; font-size: 14px; line-height: 1.6; margin-top: 4px; }
    .highlight { background-color: #ffe1ea; padding: 2px 0; font-weight: 600; }
    .stats-bar { display: flex; flex-wrap: wrap; gap: 10px; margin: 8px 0 18px 0; }
    .stat-chip { background: rgba(217, 107, 140, 0.12); color: #7b5a66; border-radius: 999px; padding: 6px 12px; font-size: 13px; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    """Load only BM25-based engine (no heavy imports)."""
    from src.indexer import InvertedIndex
    from src.ranker import BM25Ranker
    from src.searcher import SearchEngine

    index_path = "data/index.pkl"
    if not os.path.exists(index_path):
        return None

    index = InvertedIndex()
    index.load(index_path)

    bm25_ranker = BM25Ranker(index)

    return SearchEngine(
        index,
        bm25_ranker,
        reranker=None,
        tfidf_ranker=None,
        query_expander=None,
        dense_retriever=None,
        splade_retriever=None,
    )


def highlight_text(text, query, max_length=300):
    if not text:
        return ""
    query_terms = set(re.findall(r"\b\w+\b", query.lower()))
    if not query_terms:
        return text[:max_length] + ("..." if len(text) > max_length else "")

    import re as _re
    sentences = _re.split(r"[.!?]\s+", text)
    best_sentence, best_score = "", 0
    for sentence in sentences:
        words = _re.findall(r"\b\w+\b", sentence.lower())
        score = sum(1 for w in words if w in query_terms)
        if score > best_score:
            best_score, best_sentence = score, sentence
    if not best_sentence:
        best_sentence = text[:200]

    words_pattern = _re.compile(r"\b\w+\b", _re.IGNORECASE)
    def highlight_word(m):
        w = m.group(0)
        return f'<span class="highlight">{w}</span>' if w.lower() in query_terms else w
    highlighted = words_pattern.sub(highlight_word, best_sentence)

    plain = _re.sub(r"<[^>]+>", "", highlighted)
    if len(plain) > max_length:
        highlighted = highlighted[:max_length] + "..."
    return highlighted


def extract_title(doc_text, query):
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
            return title if len(title) <= 100 else title[:100] + "..."
    return first_sentence if len(first_sentence) <= 100 else first_sentence[:100] + "..."


def main():
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    if "results_per_page" not in st.session_state:
        st.session_state.results_per_page = 10

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

    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(
                "<div style=\"text-align: center; margin-bottom: 30px;\">"
                "<div style=\"font-size: 90px; font-weight: 800; font-style: italic; color: #ff5c8a; letter-spacing: 2px;\">SAP</div>"
                "<div style=\"font-size: 20px; color: #ff8fb0; margin-top: 6px;\">BM25 Only</div>"
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
                st.caption("Method: BM25")
                st.markdown("<br>", unsafe_allow_html=True)
                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                with col_btn2:
                    search_clicked = st.form_submit_button("Search", use_container_width=True, type="primary")

            query = query_input.strip() if query_input else ""

            if st.session_state.search_results:
                if st.button("Clear results", use_container_width=True):
                    st.session_state.search_results = None
                    st.session_state.current_page = 1
                    st.rerun()

    with st.sidebar:
        st.header("Index Stats")
        st.markdown(f"- Documents: {engine.index.total_docs:,}")
        st.markdown(f"- Vocabulary: {len(engine.index.posting_list):,}")
        st.markdown(f"- Avg doc length: {engine.index.avg_doc_len:.1f}")

        st.markdown("---")
        st.header("Score View")
        score_norm_mode = st.selectbox(
            "Normalization",
            ["Relative %", "Min-Max (0-1)", "Z-score"],
            index=0,
            help="How to display the secondary normalized score pill",
        )

        st.markdown("---")
        st.header("Filters")
        min_rel_score = st.slider(
            "Min relative score (%)",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
            help="Filter out results below this relative score (normalized by top score)",
        )
        include_terms = st.text_input(
            "Include terms (any)",
            value="",
            help="If set, keep results containing any of these terms",
        ).strip()
        exclude_terms = st.text_input(
            "Exclude terms",
            value="",
            help="If set, drop results containing any of these terms",
        ).strip()

        # Doc length range filter (in tokens)
        if engine.index.doc_len:
            min_len = min(engine.index.doc_len.values())
            max_len = max(engine.index.doc_len.values())
            doc_len_range = st.slider(
                "Document length (tokens)",
                min_value=int(min_len),
                max_value=int(max_len),
                value=(int(min_len), int(max_len)),
                step=1,
            )
        else:
            doc_len_range = (0, 10**9)

    if query and search_clicked:
        status = st.status("Searching...", expanded=False) if hasattr(st, "status") else None
        with st.spinner("Searching..."):
            start_time = time.time()
            try:
                result = engine.search(query, top_k=100, method="bm25", use_reranker=False)
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

        # Prepare filters
        include_set = {t.strip().lower() for t in include_terms.split() if t.strip()} if include_terms else set()
        exclude_set = {t.strip().lower() for t in exclude_terms.split() if t.strip()} if exclude_terms else set()

        original_total = len(result["results"])
        scores_all = [r["score"] for r in result["results"]]
        max_score = max(scores_all, default=1.0) or 1.0
        min_score = min(scores_all, default=0.0)
        mean_score = (sum(scores_all) / len(scores_all)) if scores_all else 0.0
        if scores_all and len(scores_all) > 1:
            var = sum((s - mean_score) ** 2 for s in scores_all) / len(scores_all)
            std_score = math.sqrt(var) if var > 0 else 0.0
        else:
            std_score = 0.0

        # Apply filters on the full list first
        filtered = []
        for r in result["results"]:
            doc_id = r["doc_id"]
            score = r["score"]
            rel_pct = min(100.0, (score / max_score) * 100.0) if max_score > 0 else 0.0
            if rel_pct < min_rel_score:
                continue
            # doc length filter
            dlen = engine.index.doc_len.get(doc_id, 0)
            if not (doc_len_range[0] <= dlen <= doc_len_range[1]):
                continue
            # include/exclude terms
            if include_set or exclude_set:
                text_low = engine.get_document(doc_id).lower()
                if include_set and not any(term in text_low for term in include_set):
                    continue
                if exclude_set and any(term in text_low for term in exclude_set):
                    continue
            filtered.append(r)

        st.markdown(
            f"""
        <div class=\"stats-bar\">\n            <span class=\"stat-chip\">Results: {len(filtered):,} / {original_total:,}</span>\n            <span class=\"stat-chip\">Time: {st.session_state.search_time:.3f}s</span>\n            <span class=\"stat-chip\">Method: BM25</span>\n        </div>
        """,
            unsafe_allow_html=True,
        )

        if not filtered:
            st.warning("No results match current filters.")
        else:
            total_results = len(filtered)
            total_pages = (total_results - 1) // st.session_state.results_per_page + 1
            start_idx = (st.session_state.current_page - 1) * st.session_state.results_per_page
            end_idx = start_idx + st.session_state.results_per_page
            page_results = filtered[start_idx:end_idx]

            for r in page_results:
                doc_id = r["doc_id"]
                score = r["score"]
                snippet = r["snippet"]
                full_text = engine.get_document(doc_id)
                score_pct = min(100.0, (score / max_score) * 100.0) if max_score > 0 else 0.0

                title = extract_title(full_text, result["query"])
                highlighted_title = highlight_text(title, result["query"], max_length=150)
                highlighted_snippet = highlight_text(snippet, result["query"])

                # Choose normalized pill based on selection
                if score_norm_mode == "Min-Max (0-1)":
                    denom = (max_score - min_score)
                    mm = ((score - min_score) / denom) if denom > 0 else 0.0
                    norm_pill = f"<span class=\"score-pill\">MinMax {mm:.3f}</span>"
                elif score_norm_mode == "Z-score":
                    z = ((score - mean_score) / std_score) if std_score > 0 else 0.0
                    norm_pill = f"<span class=\"score-pill\">Z {z:.3f}</span>"
                else:
                    norm_pill = f"<span class=\"score-pill\">Rel {score_pct:.1f}%</span>"

                st.markdown(
                    f"""
                <div class=\"result-card\">\n                    <h3 class=\"result-title\">{highlighted_title}</h3>\n                    <div class=\"result-meta\">\n                        <span class=\"rank-badge\">Rank {r['rank']}</span>\n                        <span class=\"doc-id\">{doc_id}</span>\n                        <span class=\"score-pill\">Score (BM25) {score:.4f}</span>\n                        {norm_pill}\n                    </div>\n                    <div class=\"snippet\">\n                        {highlighted_snippet if highlighted_snippet else (highlight_text(full_text[:200], result['query']) if full_text else snippet)}\n                    </div>\n                    <div class=\"score-bar\" title=\"Relative score\">\n                        <div class=\"score-bar-fill\" style=\"width: {score_pct:.1f}%;\"></div>\n                    </div>\n                </div>
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
                        f"<div style=\"text-align: center; padding: 10px;\">Page {st.session_state.current_page} / {total_pages}</div>",
                        unsafe_allow_html=True,
                    )
                with pagination_cols[2]:
                    if st.button("Next", disabled=(st.session_state.current_page >= total_pages)):
                        st.session_state.current_page += 1
                        st.rerun()

            # Results per page selector
            st.markdown("---")
            new_per_page = st.selectbox(
                "Results per page",
                [5, 10, 20, 30, 50],
                index=[5, 10, 20, 30, 50].index(st.session_state.results_per_page),
                key="per_page_selector_bm25",
            )
            if new_per_page != st.session_state.results_per_page:
                st.session_state.results_per_page = new_per_page
                st.session_state.current_page = 1
                st.rerun()


if __name__ == "__main__":
    main()
