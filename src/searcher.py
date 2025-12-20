"""
검색기 모듈
BM25 + Reranker 통합 검색 인터페이스
"""
import re
from .tokenizer import Tokenizer


class SearchEngine:
    """
    통합 검색 엔진
    BM25 기본 검색 + 선택적 Reranking
    """
    
    def __init__(self, index, ranker, reranker=None):
        self.index = index
        self.ranker = ranker
        self.reranker = reranker
        self.tokenizer = Tokenizer()
    
    def search(self, query, top_k=10, use_reranker=False, num_candidates=100):
        """
        검색 실행
        
        Args:
            query: 검색 쿼리
            top_k: 반환 결과 수
            use_reranker: 리랭커 사용 여부
            num_candidates: BM25 후보 수 (리랭킹용)
        
        Returns:
            {
                'query': str,
                'method': str,
                'results': [{'rank', 'doc_id', 'score', 'snippet'}, ...]
            }
        """
        # BM25 검색
        bm25_results = self.ranker.score(query, top_k=num_candidates)
        
        if not bm25_results:
            return {'query': query, 'method': 'BM25', 'results': []}
        
        # 리랭킹 적용
        if use_reranker and self.reranker:
            final = self.reranker.rerank(
                query, bm25_results, self.index.doc_store, top_k=top_k
            )
            method = "BM25 + Reranker"
        else:
            final = bm25_results[:top_k]
            method = "BM25"
        
        # 결과 포맷팅
        results = []
        for rank, (doc_id, score) in enumerate(final, 1):
            snippet = self._extract_snippet(doc_id, query)
            results.append({
                'rank': rank,
                'doc_id': doc_id,
                'score': score,
                'snippet': snippet
            })
        
        return {
            'query': query,
            'method': method,
            'results': results
        }
    
    def _extract_snippet(self, doc_id, query, max_len=200):
        """쿼리 관련 스니펫 추출"""
        text = self.index.get_document(doc_id)
        if not text:
            return ""
        
        query_terms = set(self.tokenizer.tokenize(query))
        words = text.split()
        
        # 쿼리 단어가 가장 많이 나오는 윈도우 찾기
        best_start = 0
        best_count = 0
        window = 25
        
        for i in range(len(words)):
            count = 0
            for j in range(i, min(i + window, len(words))):
                word_clean = words[j].lower().strip('.,!?;:"\'')
                if word_clean in query_terms:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_start = i
        
        # 스니펫 생성
        start = max(0, best_start - 3)
        end = min(len(words), best_start + window)
        snippet = ' '.join(words[start:end])
        
        if len(snippet) > max_len:
            snippet = snippet[:max_len] + "..."
        
        return snippet
    
    def get_document(self, doc_id):
        """전체 문서 반환"""
        return self.index.get_document(doc_id)
