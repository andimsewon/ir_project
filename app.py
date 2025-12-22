"""
구글 스타일 검색 엔진 UI
심플하고 창의적인 인터페이스
"""
import streamlit as st
import time
import re
import os
import importlib.util
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일
st.markdown("""
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

    /* 검색 입력 */
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

    /* 결과 카드 */
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

    /* 하이라이트 */
    .highlight {
        background-color: #ffe1ea;
        padding: 2px 0;
        font-weight: 600;
    }

    /* 검색 통계 */
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

    /* 페이지네이션 */
    .pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 30px 0;
    }

    /* 헤더 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """검색 엔진 로드 (캐시)"""
    from src.indexer import InvertedIndex
    from src.ranker import BM25Ranker
    from src.tfidf_ranker import TFIDFRanker
    from src.reranker import CrossEncoderReranker
    from src.query_expander import QueryExpander
    from src.searcher import SearchEngine
    from src.splade_retriever import SpladeRetriever
    
    index_path = "data/index.pkl"
    if not os.path.exists(index_path):
        return None
    
    index = InvertedIndex()
    index.load(index_path)
    
    bm25_ranker = BM25Ranker(index)
    tfidf_ranker = TFIDFRanker(index)
    
    # 최적화된 리랭커 (균형잡힌 성능)
    reranker = CrossEncoderReranker(model_size="balanced")
    
    # 임베딩 기반 쿼리 확장 (선택적)
    query_expander = QueryExpander(index, use_embedding=False)  # False로 설정하면 빠름
    
    splade_path = "data/splade_index.pt"
    if not os.path.exists(splade_path):
        return None
    device = "dml" if importlib.util.find_spec("torch_directml") is not None else None
    splade_retriever = SpladeRetriever(device=device)
    splade_retriever.load(splade_path)

    return SearchEngine(index, bm25_ranker, reranker, tfidf_ranker, query_expander, splade_retriever=splade_retriever)


def highlight_text(text, query, max_length=300):
    """쿼리 단어 하이라이트 및 스니펫 생성"""
    if not text:
        return ""
    
    # 쿼리 단어 추출 (소문자로 정규화)
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    
    if not query_terms:
        # 쿼리 단어가 없으면 원본 텍스트 반환 (길이 제한)
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    # 텍스트를 문장 단위로 분리
    sentences = re.split(r'[.!?]\s+', text)
    
    # 쿼리 단어가 가장 많이 포함된 문장 찾기
    best_sentence = ""
    best_score = 0
    
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(1 for w in words if w in query_terms)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    
    # 최선의 문장이 없으면 원본 텍스트 사용
    if not best_sentence:
        best_sentence = text[:200]
    
    # 하이라이트 적용 (원본 대소문자 유지)
    words_pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
    
    def highlight_word(match):
        word = match.group(0)
        if word.lower() in query_terms:
            return f'<span class="highlight">{word}</span>'
        return word
    
    highlighted = words_pattern.sub(highlight_word, best_sentence)
    
    # 길이 제한 (HTML 태그 제외하고 계산)
    plain_text = re.sub(r'<[^>]+>', '', highlighted)
    if len(plain_text) > max_length:
        # 하이라이트 태그를 고려하여 자르기
        truncated = ""
        tag_open = False
        for char in highlighted:
            if char == '<':
                tag_open = True
            if not tag_open:
                truncated += char
                if len(re.sub(r'<[^>]+>', '', truncated)) >= max_length:
                    break
            if char == '>':
                tag_open = False
        highlighted = truncated + "..."
    
    return highlighted


def extract_title(doc_text, query):
    """문서에서 제목 추출 (첫 문장 또는 쿼리 관련 부분)"""
    if not doc_text:
        return "Untitled Document"
    
    # 첫 문장을 제목으로 사용
    first_sentence = doc_text.split('.')[0].strip()
    
    # 쿼리 단어가 포함된 경우 해당 부분 우선
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    words = doc_text.split()
    
    for i, word in enumerate(words[:50]):  # 처음 50단어만 확인
        if word.lower().strip('.,!?;:"\'') in query_terms:
            # 해당 단어 주변을 제목으로
            start = max(0, i - 5)
            end = min(len(words), i + 15)
            title = ' '.join(words[start:end])
            if len(title) > 100:
                title = title[:100] + "..."
            return title
    
    # 기본: 첫 문장
    if len(first_sentence) > 100:
        first_sentence = first_sentence[:100] + "..."
    return first_sentence or "Document"


def main():
    # 세션 상태 초기화
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    if 'results_per_page' not in st.session_state:
        st.session_state.results_per_page = 10
    if 'filter_method' not in st.session_state:
        st.session_state.filter_method = "bm25"
    if 'method_option' not in st.session_state:
        st.session_state.method_option = "BM25"
    if 'use_reranker_opt' not in st.session_state:
        st.session_state.use_reranker_opt = False
    if 'use_expansion_opt' not in st.session_state:
        st.session_state.use_expansion_opt = False
    if 'hybrid_weight' not in st.session_state:
        st.session_state.hybrid_weight = 0.6
    if 'pending_query' not in st.session_state:
        st.session_state.pending_query = None

    if st.session_state.pending_query:
        st.session_state.search_input = st.session_state.pending_query
        st.session_state.pending_query = None
        st.session_state.search_results = None
        st.session_state.current_page = 1

    index_path = "data/index.pkl"
    splade_path = "data/splade_index.pt"
    if not os.path.exists(index_path) or not os.path.exists(splade_path):
        st.error("Required index files are missing.")
        st.code("python download_data.py\npython build_index.py\npython build_splade_index.py", language="bash")
        return

    engine = load_engine()
    
    if engine is None:
        st.error("⚠️ 인덱스를 찾을 수 없습니다. 먼저 다음 명령을 실행하세요:")
        st.code("python download_data.py\npython build_index.py\npython build_splade_index.py", language="bash")
        return
    
    # 메인 컨테이너
    with st.container():
        # 로고 및 검색창 영역
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(
                '<div style="text-align: center; margin-bottom: 30px;">'
                '<div style="font-size: 90px; font-weight: 800; font-style: italic; color: #ff5c8a; letter-spacing: 2px;">SAP</div>'
                '<div style="font-size: 20px; color: #ff8fb0; margin-top: 6px;">Search Anything Positively</div>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # 검색 입력
            query_input = st.text_input(
                "",
                value=st.session_state.get('search_input', ''),
                placeholder="검색어를 입력하세요...",
                key="search_input",
                label_visibility="collapsed"
            )
            
            # 검색 버튼 (Enter 키로도 작동)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                search_clicked = st.button("검색", use_container_width=True, type="primary")
            
            # 검색어 업데이트
            query = query_input.strip() if query_input else ""
            
            # ?? ?? (?? ??)
            st.markdown("<br>", unsafe_allow_html=True)
            filter_cols = st.columns(5)
            methods = [
                ("BM25", "bm25"),
                ("TF-IDF", "tfidf"),
                ("하이브리드", "hybrid"),
                ("리랭커", "rerank"),
                ("쿼리 확장", "expansion"),
            ]
            for i, (label, method) in enumerate(methods):
                with filter_cols[i]:
                    is_active = (
                        st.session_state.filter_method == method
                        or (method == "rerank" and st.session_state.use_reranker_opt)
                        or (method == "expansion" and st.session_state.use_expansion_opt)
                    )
                    button_label = f"[ON] {label}" if is_active else label
                    if st.button(button_label, key=f"filter_{method}", use_container_width=True):
                        st.session_state.filter_method = method
                        if method == "bm25":
                            st.session_state.method_option = "BM25"
                        elif method == "tfidf":
                            st.session_state.method_option = "TF-IDF"
                        elif method == "hybrid":
                            st.session_state.method_option = "하이브리드"
                        elif method == "rerank":
                            st.session_state.use_reranker_opt = True
                        elif method == "expansion":
                            st.session_state.use_expansion_opt = True
            active_method = st.session_state.method_option
            if st.session_state.filter_method in ["bm25", "tfidf", "hybrid"]:
                active_method = {"bm25": "BM25", "tfidf": "TF-IDF", "hybrid": "하이브리드"}[st.session_state.filter_method]
            rerank_active = st.session_state.use_reranker_opt or st.session_state.filter_method == "rerank"
            expansion_active = st.session_state.use_expansion_opt or st.session_state.filter_method == "expansion"
            chips = [
                f"방법: {active_method}",
                f"리랭커: {"ON" if rerank_active else "OFF"}",
                f"쿼리 확장: {"ON" if expansion_active else "OFF"}",
            ]
            if active_method == "하이브리드":
                chips.append(f"BM25 가중치: {st.session_state.hybrid_weight:.1f}")
            st.caption(" | ".join(chips))
    with st.sidebar:
        st.header("⚙️ 고급 설정")
        
        if engine:
            st.markdown(f"**인덱스 정보**")
            st.markdown(f"- 문서 수: {engine.index.total_docs:,}")
            st.markdown(f"- 어휘 크기: {len(engine.index.posting_list):,}")
            st.markdown(f"- 평균 문서 길이: {engine.index.avg_doc_len:.1f}")
        
        st.markdown("---")
        st.markdown("**검색 방법**")
        method_option = st.selectbox(
            "랭킹 방법",
            ["BM25", "TF-IDF", "하이브리드"],
            index=["BM25", "TF-IDF", "하이브리드"].index(st.session_state.method_option) if st.session_state.method_option in ["BM25", "TF-IDF", "하이브리드"] else 0,
            key="method_selectbox"
        )
        st.session_state.method_option = method_option
        
        use_reranker_opt = st.checkbox("리랭커 사용", value=st.session_state.use_reranker_opt, key="reranker_checkbox")
        st.session_state.use_reranker_opt = use_reranker_opt
        
        use_expansion_opt = st.checkbox("쿼리 확장", value=st.session_state.use_expansion_opt, key="expansion_checkbox")
        st.session_state.use_expansion_opt = use_expansion_opt
        
        if method_option == "하이브리드":
            hybrid_weight = st.slider("BM25 가중치", 0.0, 1.0, st.session_state.hybrid_weight, 0.1, key="hybrid_slider")
            st.session_state.hybrid_weight = hybrid_weight
    
    # 검색 실행
    if (search_clicked or query) and query.strip():
        status = st.status("검색 중...", expanded=False) if hasattr(st, "status") else None
        with st.spinner("검색 중..."):
            start_time = time.time()
            
            # 필터에 따른 검색 설정
            use_reranker = st.session_state.filter_method == "rerank" or st.session_state.use_reranker_opt
            use_expansion = st.session_state.filter_method == "expansion" or st.session_state.use_expansion_opt
            method = "bm25"
            
            if st.session_state.filter_method == "tfidf" or st.session_state.method_option == "TF-IDF":
                method = "tfidf"
            elif st.session_state.filter_method == "hybrid" or st.session_state.method_option == "하이브리드":
                method = "hybrid"
            
            result = engine.search(
                query,
                top_k=100,  # 더 많은 결과를 가져와서 페이지네이션
                method=method,
                use_reranker=use_reranker,
                use_query_expansion=use_expansion,
                hybrid_weight=st.session_state.hybrid_weight
            )
            
            elapsed = time.time() - start_time
            st.session_state.search_results = result
            st.session_state.search_time = elapsed
            st.session_state.current_page = 1
            if status:
                status.update(label=f"검색 완료 ({elapsed:.2f}s)", state="complete")
    
    # 검색 결과 표시
    if st.session_state.search_results:
        result = st.session_state.search_results
        
        # 검색 통계
        st.markdown(f"""
        <div class="stats-bar">
            <span class="stat-chip">Results: {len(result['results']):,}</span>
            <span class="stat-chip">Time: {st.session_state.search_time:.3f}s</span>
            <span class="stat-chip">Method: {result['method']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # 확장된 쿼리 표시
        if result.get('expanded_query') and result['expanded_query'] != result['query']:
            st.info(f"💡 확장된 쿼리: **{result['expanded_query']}**")
        
        if not result['results']:
            st.warning("검색 결과가 없습니다. 다른 검색어를 시도해보세요.")
            st.markdown("**추천 검색어:**")
            examples = ["machine learning", "artificial intelligence", "world war", "climate change"]
            cols = st.columns(len(examples))
            for i, ex in enumerate(examples):
                if cols[i].button(ex, key=f"ex_no_results_{i}"):
                    # 瓴€?夓柎毳??胳厴 ?來儨리랭커€?ロ晿瓿?瓴€리랭커ろ枆
                    st.session_state.pending_query = ex
                    st.rerun()
        else:
            # 페이지네이션 계산
            total_results = len(result['results'])
            total_pages = (total_results - 1) // st.session_state.results_per_page + 1
            start_idx = (st.session_state.current_page - 1) * st.session_state.results_per_page
            end_idx = start_idx + st.session_state.results_per_page
            page_results = result['results'][start_idx:end_idx]
            max_score = max((r['score'] for r in result['results']), default=1.0) or 1.0
            
            # 결과 표시
            for r in page_results:
                doc_id = r['doc_id']
                score = r['score']
                snippet = r['snippet']
                full_text = engine.get_document(doc_id)
                score_pct = min(100.0, (score / max_score) * 100.0) if max_score > 0 else 0.0
                
                # 제목 추출
                title = extract_title(full_text, result['query'])
                highlighted_title = highlight_text(title, result['query'], max_length=150)
                
                # 스니펫 하이라이트
                highlighted_snippet = highlight_text(snippet, result['query'])
                
                # 결과 카드
                st.markdown(f"""
                <div class="result-card">
                    <h3 class="result-title">{highlighted_title}</h3>
                    <div class="result-meta">
                        <span class="rank-badge">Rank {r['rank']}</span>
                        <span class="doc-id">{doc_id}</span>
                        <span class="score-pill">Score {score:.4f}</span>
                    </div>
                    <div class="snippet">
                        {highlighted_snippet if highlighted_snippet else snippet}
                    </div>
                    <div class="score-bar" title="Relative score">
                        <div class="score-bar-fill" style="width: {score_pct:.1f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 전체 문서 보기 (확장 가능)
                with st.expander("📄 전체 문서 보기", expanded=False):
                    st.text_area("", full_text[:5000], height=200, key=f"full_{doc_id}", disabled=True)
            
            # 페이지네이션
            if total_pages > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                pagination_cols = st.columns([1, 2, 1])
                
                with pagination_cols[0]:
                    if st.button("◀ 이전", disabled=(st.session_state.current_page == 1)):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                with pagination_cols[1]:
                    st.markdown(
                        f'<div style="text-align: center; padding: 10px;">'
                        f'페이지 {st.session_state.current_page} / {total_pages}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with pagination_cols[2]:
                    if st.button("다음 ▶", disabled=(st.session_state.current_page >= total_pages)):
                        st.session_state.current_page += 1
                        st.rerun()
            
            # 결과 수 조정
            st.markdown("---")
            new_per_page = st.selectbox(
                "페이지당 결과 수",
                [5, 10, 20, 30, 50],
                index=[5, 10, 20, 30, 50].index(st.session_state.results_per_page),
                key="per_page_selector"
            )
            if new_per_page != st.session_state.results_per_page:
                st.session_state.results_per_page = new_per_page
                st.session_state.current_page = 1
                st.rerun()
    


if __name__ == "__main__":
    main()
