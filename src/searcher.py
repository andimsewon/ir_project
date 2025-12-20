"""
검색기 모듈
BM25 + TF-IDF + Reranker 통합 검색 인터페이스
"""
import re
from collections import defaultdict
from .tokenizer import Tokenizer


class SearchEngine:
    """
    통합 검색 엔진
    BM25/TF-IDF/하이브리드 랭킹 + 선택적 Reranking + 쿼리 확장
    """
    
    def __init__(self, index, ranker, reranker=None, tfidf_ranker=None, query_expander=None):
        self.index = index
        self.ranker = ranker  # BM25 ranker
        self.tfidf_ranker = tfidf_ranker  # TF-IDF ranker (optional)
        self.reranker = reranker
        self.query_expander = query_expander
        self.tokenizer = Tokenizer()
    
    def search(self, query, top_k=10, method="bm25", use_reranker=False, 
               use_query_expansion=False, num_candidates=100, hybrid_weight=0.5):
        """
        검색 실행
        
        Args:
            query: 검색 쿼리
            top_k: 반환 결과 수
            method: "bm25", "tfidf", "hybrid"
            use_reranker: 리랭커 사용 여부
            use_query_expansion: 쿼리 확장 사용 여부
            num_candidates: 후보 수 (리랭킹용)
            hybrid_weight: 하이브리드 랭킹에서 BM25 가중치 (0~1)
        
        Returns:
            {
                'query': str,
                'expanded_query': str,
                'method': str,
                'results': [{'rank', 'doc_id', 'score', 'snippet'}, ...]
            }
        """
        original_query = query
        
        # 쿼리 확장
        if use_query_expansion and self.query_expander:
            query = self.query_expander.expand(query, method="synonym")
            expanded_query = query
        else:
            expanded_query = original_query
        
        # 랭킹 방법 선택
        if method == "bm25":
            ranked_results = self.ranker.score(query, top_k=num_candidates)
            method_name = "BM25"
        elif method == "tfidf":
            if not self.tfidf_ranker:
                raise ValueError("TF-IDF ranker not provided")
            ranked_results = self.tfidf_ranker.score(query, top_k=num_candidates)
            method_name = "TF-IDF"
        elif method == "hybrid":
            if not self.tfidf_ranker:
                raise ValueError("TF-IDF ranker not provided for hybrid ranking")
            ranked_results = self._hybrid_score(query, hybrid_weight, num_candidates)
            method_name = f"Hybrid (BM25:{hybrid_weight:.1f} + TF-IDF:{1-hybrid_weight:.1f})"
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if not ranked_results:
            return {
                'query': original_query,
                'expanded_query': expanded_query,
                'method': method_name,
                'results': []
            }
        
        # 리랭킹 적용
        if use_reranker and self.reranker:
            final = self.reranker.rerank(
                original_query, ranked_results, self.index.doc_store, top_k=top_k
            )
            method_name += " + Reranker"
        else:
            final = ranked_results[:top_k]
        
        # 결과 포맷팅
        results = []
        for rank, (doc_id, score) in enumerate(final, 1):
            snippet = self._extract_snippet(doc_id, original_query)
            results.append({
                'rank': rank,
                'doc_id': doc_id,
                'score': score,
                'snippet': snippet
            })
        
        return {
            'query': original_query,
            'expanded_query': expanded_query if use_query_expansion else None,
            'method': method_name,
            'results': results
        }
    
    def _hybrid_score(self, query, bm25_weight, top_k=100):
        """
        하이브리드 랭킹: BM25와 TF-IDF 점수 결합
        
        Args:
            query: 검색 쿼리
            bm25_weight: BM25 가중치 (0~1)
            top_k: 반환할 결과 수
        
        Returns: [(doc_id, hybrid_score), ...]
        """
        bm25_results = self.ranker.score(query, top_k=top_k)
        tfidf_results = self.tfidf_ranker.score(query, top_k=top_k)
        
        # 점수 정규화를 위한 최대값 찾기
        bm25_max = max((score for _, score in bm25_results), default=1.0)
        tfidf_max = max((score for _, score in tfidf_results), default=1.0)
        
        # 문서별 점수 합산
        doc_scores = defaultdict(float)
        
        # BM25 점수 추가
        for doc_id, score in bm25_results:
            normalized_score = score / bm25_max if bm25_max > 0 else 0
            doc_scores[doc_id] += bm25_weight * normalized_score
        
        # TF-IDF 점수 추가
        for doc_id, score in tfidf_results:
            normalized_score = score / tfidf_max if tfidf_max > 0 else 0
            doc_scores[doc_id] += (1 - bm25_weight) * normalized_score
        
        # 점수순 정렬
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
    
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
