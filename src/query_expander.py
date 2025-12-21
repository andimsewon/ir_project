"""
쿼리 확장 모듈
임베딩 기반 쿼리 확장 (LLM 활용)
"""
from collections import Counter
import math
from .tokenizer import Tokenizer
import warnings
warnings.filterwarnings('ignore')


class QueryExpander:
    """
    쿼리 확장기
    - 동의어 확장 (간단한 사전 기반)
    - 관련어 확장 (인덱스 통계 기반)
    - 임베딩 기반 확장 (LLM 활용, 선택적)
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
        "learn": ["study", "understand", "grasp"],
        "study": ["learn", "research", "examine"],
        "war": ["conflict", "battle", "combat"],
        "climate": ["weather", "environment", "atmosphere"],
    }
    
    def __init__(self, index=None, use_embedding=False):
        """
        Args:
            index: InvertedIndex 인스턴스
            use_embedding: 임베딩 기반 확장 사용 여부
        """
        self.index = index
        self.tokenizer = Tokenizer()
        self.use_embedding = use_embedding
        self.embedding_model = None
        
        if use_embedding:
            try:
                from sentence_transformers import SentenceTransformer
                # 작고 빠른 임베딩 모델
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                print(f"[QueryExpander] Loading embedding model: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                print(f"[QueryExpander] Embedding model loaded")
            except Exception as e:
                print(f"[QueryExpander] Warning: Could not load embedding model: {e}")
                print(f"[QueryExpander] Falling back to dictionary-based expansion")
                self.use_embedding = False
    
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
                if syn.lower() not in [t.lower() for t in expanded_terms]:
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
                    doc_freq = self.index.get_doc_freq(other_term)
                    total_docs = self.index.total_docs or 0
                    if total_docs > 0:
                        idf = math.log(total_docs / (doc_freq + 1))
                        score = cooccur_count * idf
                        cooccurrence_scores[other_term] += score
        
        # 상위 k개 단어 추가
        expanded_terms = list(query_terms)
        for term, _ in cooccurrence_scores.most_common(top_k):
            expanded_terms.append(term)
        
        return ' '.join(expanded_terms)
    
    def expand_with_embedding(self, query, top_k=3):
        """
        임베딩 기반 쿼리 확장
        쿼리와 유사한 의미의 단어를 임베딩 공간에서 찾아 확장
        
        Args:
            query: 원본 쿼리
            top_k: 추가할 단어 수
        
        Returns: 확장된 쿼리
        """
        if not self.embedding_model or not self.index:
            return query
        
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            
            # 인덱스의 주요 단어들 중에서 유사한 단어 찾기
            # (빈도가 높은 단어들만 고려하여 계산량 감소)
            candidate_terms = []
            query_terms = set(self.tokenizer.tokenize(query))
            
            # 상위 빈도 단어들만 후보로 선택 (성능 최적화)
            term_freqs = [(term, len(posting)) for term, posting in self.index.posting_list.items()]
            term_freqs.sort(key=lambda x: x[1], reverse=True)
            top_terms = [term for term, _ in term_freqs[:1000]]  # 상위 1000개만
            
            for term in top_terms:
                if term.lower() not in [t.lower() for t in query_terms]:
                    candidate_terms.append(term)
            
            if not candidate_terms:
                return query
            
            # 후보 단어들의 임베딩 계산
            term_embeddings = self.embedding_model.encode(
                candidate_terms[:100],  # 최대 100개만 (성능 최적화)
                convert_to_tensor=True
            )
            
            # 코사인 유사도 계산
            from torch.nn.functional import cosine_similarity
            similarities = cosine_similarity(query_embedding.unsqueeze(0), term_embeddings)[0]
            
            # 상위 k개 선택
            top_indices = similarities.topk(min(top_k, len(candidate_terms))).indices
            expanded_terms = list(query_terms)
            
            for idx in top_indices:
                term = candidate_terms[idx]
                if term.lower() not in [t.lower() for t in expanded_terms]:
                    expanded_terms.append(term)
            
            return ' '.join(expanded_terms)
        except Exception as e:
            print(f"[QueryExpander] Embedding expansion failed: {e}")
            return query
    
    def expand(self, query, method="synonym", **kwargs):
        """
        통합 쿼리 확장
        
        Args:
            query: 원본 쿼리
            method: "synonym", "cooccurrence", "embedding", "hybrid"
            **kwargs: 각 방법에 대한 추가 파라미터
        
        Returns: 확장된 쿼리
        """
        if method == "synonym":
            return self.expand_with_synonyms(query, **kwargs)
        elif method == "cooccurrence":
            return self.expand_with_cooccurrence(query, **kwargs)
        elif method == "embedding":
            return self.expand_with_embedding(query, **kwargs)
        elif method == "hybrid":
            # 하이브리드: 동의어 + 공출현
            expanded = self.expand_with_synonyms(query, **kwargs)
            expanded = self.expand_with_cooccurrence(expanded, top_k=2)
            return expanded
        else:
            return query
