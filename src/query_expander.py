"""
쿼리 확장 모듈
동의어 및 관련어 기반 쿼리 확장
"""
from collections import Counter
from .tokenizer import Tokenizer


class QueryExpander:
    """
    쿼리 확장기
    - 동의어 확장 (간단한 사전 기반)
    - 관련어 확장 (인덱스 통계 기반)
    """
    
    # 간단한 동의어 사전 (예시)
    SYNONYMS = {
        "car": ["automobile", "vehicle", "auto"],
        "automobile": ["car", "vehicle", "auto"],
        "vehicle": ["car", "automobile", "auto"],
        "computer": ["pc", "machine", "device"],
        "pc": ["computer", "machine"],
        "movie": ["film", "cinema", "picture"],
        "film": ["movie", "cinema", "picture"],
        "big": ["large", "huge", "enormous"],
        "large": ["big", "huge", "enormous"],
        "small": ["tiny", "little", "mini"],
        "good": ["great", "excellent", "fine"],
        "bad": ["poor", "terrible", "awful"],
    }
    
    def __init__(self, index=None):
        self.index = index
        self.tokenizer = Tokenizer()
    
    def expand_with_synonyms(self, query, max_expansions=2):
        """
        동의어로 쿼리 확장
        
        Args:
            query: 원본 쿼리
            max_expansions: 각 단어당 최대 확장 수
        
        Returns: 확장된 쿼리 (원본 + 동의어)
        """
        terms = self.tokenizer.tokenize(query)
        expanded_terms = list(terms)
        
        for term in terms:
            synonyms = self.SYNONYMS.get(term.lower(), [])
            # 최대 max_expansions개만 추가
            for syn in synonyms[:max_expansions]:
                if syn not in expanded_terms:
                    expanded_terms.append(syn)
        
        return ' '.join(expanded_terms)
    
    def expand_with_cooccurrence(self, query, top_k=3):
        """
        공출현 기반 쿼리 확장
        쿼리 단어와 함께 자주 나타나는 단어를 찾아 확장
        
        Args:
            query: 원본 쿼리
            top_k: 추가할 단어 수
        
        Returns: 확장된 쿼리
        """
        if not self.index:
            return query
        
        query_terms = set(self.tokenizer.tokenize(query))
        cooccurrence_scores = Counter()
        
        # 각 쿼리 단어의 posting list를 확인
        for term in query_terms:
            posting = self.index.get_posting(term)
            doc_ids_with_term = {doc_id for doc_id, _ in posting}
            
            # 해당 문서들에서 함께 나타나는 다른 단어 찾기
            for other_term, other_posting in self.index.posting_list.items():
                if other_term in query_terms:
                    continue
                
                doc_ids_with_other = {doc_id for doc_id, _ in other_posting}
                # 공출현 문서 수
                cooccur_count = len(doc_ids_with_term & doc_ids_with_other)
                
                if cooccur_count > 0:
                    # IDF 가중치 적용
                    idf = self.index.get_doc_freq(other_term)
                    if idf > 0:
                        score = cooccur_count / idf
                        cooccurrence_scores[other_term] += score
        
        # 상위 k개 단어 추가
        expanded_terms = list(query_terms)
        for term, _ in cooccurrence_scores.most_common(top_k):
            expanded_terms.append(term)
        
        return ' '.join(expanded_terms)
    
    def expand(self, query, method="synonym", **kwargs):
        """
        통합 쿼리 확장
        
        Args:
            query: 원본 쿼리
            method: "synonym" 또는 "cooccurrence"
            **kwargs: 각 방법에 대한 추가 파라미터
        
        Returns: 확장된 쿼리
        """
        if method == "synonym":
            return self.expand_with_synonyms(query, **kwargs)
        elif method == "cooccurrence":
            return self.expand_with_cooccurrence(query, **kwargs)
        else:
            return query

