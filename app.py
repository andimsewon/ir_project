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
    from src.tfidf_ranker import TFIDFRanker
    from src.reranker import CrossEncoderReranker
    from src.query_expander import QueryExpander
    from src.searcher import SearchEngine
    
    index_path = "data/index.pkl"
    if not os.path.exists(index_path):
        return None
    
    index = InvertedIndex()
    index.load(index_path)
    
    bm25_ranker = BM25Ranker(index)
    tfidf_ranker = TFIDFRanker(index)
    reranker = CrossEncoderReranker()
    query_expander = QueryExpander(index)
    
    return SearchEngine(index, bm25_ranker, reranker, tfidf_ranker, query_expander)


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
    st.caption("wikir/en1k dataset | BM25 + TF-IDF + Cross-Encoder Reranker")
    
    # 검색 히스토리 초기화
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    engine = load_engine()
    
    if engine is None:
        st.error("Index not found. Run these commands first:")
        st.code("python download_data.py\npython build_index.py")
        return
    
    # 사이드바
    st.sidebar.header("Settings")
    
    ranking_method = st.sidebar.selectbox(
        "Ranking Method",
        ["BM25", "TF-IDF", "Hybrid (BM25 + TF-IDF)"],
        index=0
    )
    
    use_reranker = st.sidebar.checkbox("Use Reranker", value=False)
    use_query_expansion = st.sidebar.checkbox("Query Expansion", value=False)
    
    num_results = st.sidebar.slider("Results per page", 5, 30, 10)
    
    # 하이브리드 가중치 설정
    if ranking_method == "Hybrid (BM25 + TF-IDF)":
        hybrid_weight = st.sidebar.slider("BM25 Weight", 0.0, 1.0, 0.5, 0.1)
    else:
        hybrid_weight = 0.5
    
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
    
    # 검색 히스토리 표시
    if st.session_state.search_history:
        with st.expander("📜 Search History", expanded=False):
            for idx, hist in enumerate(reversed(st.session_state.search_history[-10:])):
                col1, col2 = st.columns([4, 1])
                col1.text(f"{hist['query']} ({hist['method']})")
                if col2.button("🔍", key=f"hist_{idx}"):
                    query = hist['query']
                    search_btn = True
    
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
        # 메서드 매핑
        method_map = {
            "BM25": "bm25",
            "TF-IDF": "tfidf",
            "Hybrid (BM25 + TF-IDF)": "hybrid"
        }
        search_method = method_map[ranking_method]
        
        with st.spinner("Searching..."):
            start = time.time()
            result = engine.search(
                query, 
                top_k=num_results,
                method=search_method,
                use_reranker=use_reranker,
                use_query_expansion=use_query_expansion,
                hybrid_weight=hybrid_weight
            )
            elapsed = time.time() - start
        
        # 검색 히스토리에 추가
        st.session_state.search_history.append({
            'query': query,
            'method': result['method'],
            'num_results': len(result['results']),
            'elapsed': elapsed
        })
        
        st.markdown("---")
        st.markdown(f"### Results ({len(result['results'])} found, {elapsed:.3f}s)")
        st.markdown(f"**Method:** `{result['method']}`")
        
        if result.get('expanded_query') and result['expanded_query'] != query:
            st.info(f"**Expanded Query:** {result['expanded_query']}")
        
        # 쿼리 분석
        with st.expander("📊 Query Analysis", expanded=False):
            query_terms = engine.tokenizer.tokenize(query)
            if query_terms:
                st.markdown("**Query Terms:**")
                term_info = []
                for term in query_terms:
                    df = engine.index.get_doc_freq(term)
                    term_info.append({
                        'Term': term,
                        'Document Frequency': f"{df:,}",
                        'IDF': f"{engine.ranker._calc_idf(term):.4f}"
                    })
                st.table(term_info)
        
        if not result['results']:
            st.warning("No results found.")
            return
        
        # 페이지네이션
        total_results = len(result['results'])
        if 'page' not in st.session_state:
            st.session_state.page = 1
        
        if total_results > num_results:
            total_pages = (total_results - 1) // num_results + 1
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("◀ Previous", disabled=(st.session_state.page == 1)):
                    st.session_state.page -= 1
            with col2:
                st.markdown(f"**Page {st.session_state.page} of {total_pages}**", 
                           help="Use Previous/Next buttons to navigate")
            with col3:
                if st.button("Next ▶", disabled=(st.session_state.page >= total_pages)):
                    st.session_state.page += 1
            
            start_idx = (st.session_state.page - 1) * num_results
            end_idx = start_idx + num_results
            display_results = result['results'][start_idx:end_idx]
        else:
            display_results = result['results']
            st.session_state.page = 1
        
        # 결과 표시
        for r in display_results:
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
