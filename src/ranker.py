"""
랭커 모듈
BM25 스코어링 알고리즘 구현
"""
import math
from collections import defaultdict

from .tokenizer import Tokenizer


class BM25Ranker:
    """
    BM25 랭킹 알고리즘
    
    Score(D,Q) = sum over q in Q of:
        IDF(q) * (f(q,D) * (k1 + 1)) / (f(q,D) + k1 * (1 - b + b * |D| / avgdl))
    
    IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
    """
    
    def __init__(self, index, k1=1.5, b=0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.tokenizer = Tokenizer()
    
    def _calc_idf(self, term):
        """IDF 계산"""
        N = self.index.total_docs
        df = self.index.get_doc_freq(term)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def _calc_term_score(self, term, tf, doc_len):
        """단일 term의 BM25 점수 계산"""
        idf = self._calc_idf(term)
        
        avgdl = self.index.avg_doc_len
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        
        return idf * (numerator / denominator)
    
    def score(self, query, top_k=100):
        """
        쿼리에 대해 문서들의 BM25 점수 계산
        
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
                doc_scores[doc_id] += self._calc_term_score(term, tf, doc_len)
        
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked[:top_k]
