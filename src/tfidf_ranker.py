"""
TF-IDF 랭커 모듈
TF-IDF 스코어링 알고리즘 구현
"""
import math
from collections import defaultdict

from .tokenizer import Tokenizer


class TFIDFRanker:
    """
    TF-IDF 랭킹 알고리즘
    
    TF-IDF(t,d) = TF(t,d) × IDF(t)
    
    TF(t,d) = (단어 t가 문서 d에서 나타난 횟수) / (문서 d의 총 단어 수)
    IDF(t) = log(N / DF(t))
    """
    
    def __init__(self, index):
        self.index = index
        self.tokenizer = Tokenizer()
    
    def _calc_idf(self, term):
        """IDF 계산"""
        N = self.index.total_docs
        df = self.index.get_doc_freq(term)
        if df == 0:
            return 0.0
        return math.log(N / df)
    
    def _calc_tf(self, term_freq, doc_len):
        """TF 계산 (정규화된 빈도)"""
        if doc_len == 0:
            return 0.0
        return term_freq / doc_len
    
    def _calc_tfidf(self, term, tf, doc_len):
        """TF-IDF 점수 계산"""
        idf = self._calc_idf(term)
        tf_score = self._calc_tf(tf, doc_len)
        return tf_score * idf
    
    def score(self, query, top_k=100):
        """
        쿼리에 대해 문서들의 TF-IDF 점수 계산
        
        Returns: [(doc_id, score), ...] 점수 내림차순
        """
        query_terms = self.tokenizer.tokenize(query)
        
        if not query_terms:
            return []
        
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            posting = self.index.get_posting(term)
            
            for doc_id, tf in posting:
                doc_len = self.index.doc_len[doc_id]
                doc_scores[doc_id] += self._calc_tfidf(term, tf, doc_len)
        
        # 점수순 정렬
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]

