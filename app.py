"""
웹 UI (Streamlit)
검색 데모 페이지
"""
import streamlit as st
import time
import re
import os

st.set_page_config(page_title="Search Engine Demo", page_icon="🔍", layout="wide")


@st.cache_resource
def load_engine():
    """검색 엔진 로드 (캐시)"""
    from src.indexer import InvertedIndex
    from src.ranker import BM25Ranker
    from src.reranker import CrossEncoderReranker
    from src.searcher import SearchEngine
    
    index_path = "data/index.pkl"
    if not os.path.exists(index_path):
        return None
    
    index = InvertedIndex()
    index.load(index_path)
    
    ranker = BM25Ranker(index)
    reranker = CrossEncoderReranker()
    
    return SearchEngine(index, ranker, reranker)


def highlight(text, query):
    """쿼리 단어 하이라이트"""
    words = set(re.findall(r'\w+', query.lower()))
    
    def replace(m):
        w = m.group(0)
        if w.lower() in words:
            return f"**{w}**"
        return w
    
    return re.sub(r'\b\w+\b', replace, text)


def main():
    st.title("🔍 Information Retrieval Search Engine")
    st.caption("wikir/en1k dataset | BM25 + Cross-Encoder Reranker")
    
    engine = load_engine()
    
    if engine is None:
        st.error("Index not found. Run these commands first:")
        st.code("python download_data.py\npython build_index.py")
        return
    
    # 사이드바
    st.sidebar.header("Settings")
    
    method = st.sidebar.radio(
        "Search Method",
        ["BM25", "BM25 + Reranker"]
    )
    
    num_results = st.sidebar.slider("Results", 5, 30, 10)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Index Info**")
    st.sidebar.markdown(f"- Documents: {engine.index.total_docs:,}")
    st.sidebar.markdown(f"- Terms: {len(engine.index.posting_list):,}")
    
    # 검색 입력
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Search", placeholder="Enter query...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("Search", use_container_width=True)
    
    # 예시 쿼리
    st.markdown("**Examples:**")
    examples = ["machine learning", "world war II", "climate change", "python programming"]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            query = ex
            search_btn = True
    
    # 검색 실행
    if search_btn and query:
        use_reranker = (method == "BM25 + Reranker")
        
        with st.spinner("Searching..."):
            start = time.time()
            result = engine.search(query, top_k=num_results, use_reranker=use_reranker)
            elapsed = time.time() - start
        
        st.markdown("---")
        st.markdown(f"### Results ({len(result['results'])} found, {elapsed:.3f}s)")
        st.markdown(f"Method: `{result['method']}`")
        
        if not result['results']:
            st.warning("No results found.")
        
        for r in result['results']:
            with st.container():
                c1, c2 = st.columns([6, 1])
                c1.markdown(f"**#{r['rank']} Doc {r['doc_id']}**")
                c2.markdown(f"`{r['score']:.4f}`")
                
                snippet = highlight(r['snippet'], query)
                st.markdown(snippet)
                
                with st.expander("Full document"):
                    full = engine.get_document(r['doc_id'])
                    st.text_area("", full[:3000], height=150, key=f"doc_{r['doc_id']}")
                
                st.markdown("---")


if __name__ == "__main__":
    main()
