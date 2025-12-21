"""
리랭커 모듈
Cross-Encoder 기반 재순위화
최적화된 모델 선택 및 성능 향상
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class CrossEncoderReranker:
    """
    Cross-Encoder 리랭커
    BM25 결과를 신경망 모델로 재순위화
    
    모델 옵션:
    - BGE-reranker-base: 최신 모델, 좋은 성능 (권장)
    - ms-marco-MiniLM-L-6-v2: 작고 빠름 (기본값)
    - ms-marco-MiniLM-L-12-v2: 더 큰 버전, 더 좋은 성능
    """
    
    # 모델 옵션 (성능 순서)
    MODEL_OPTIONS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 가장 빠름, 작은 메모리
        "balanced": "BAAI/bge-reranker-base",  # 균형잡힌 성능 (권장)
        "best": "BAAI/bge-reranker-v2-m3",  # 최고 성능, 더 많은 메모리 필요
    }
    
    DEFAULT_MODEL = "balanced"  # 기본 모델
    
    def __init__(self, device=None, model_size="balanced"):
        """
        Args:
            device: "cpu", "cuda", "mps" 또는 None (자동 감지)
            model_size: "fast", "balanced", "best"
        """
        # device 설정
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # 모델 선택
        if model_size in self.MODEL_OPTIONS:
            self.model_name = self.MODEL_OPTIONS[model_size]
        else:
            self.model_name = self.MODEL_OPTIONS[self.DEFAULT_MODEL]
        
        print(f"[Reranker] Loading model: {self.model_name}")
        print(f"[Reranker] Device: {self.device}")
        print(f"[Reranker] Model size: {model_size}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 모델 최적화 (CPU에서 더 빠르게)
            if self.device == "cpu":
                # CPU 최적화
                try:
                    import torch.jit
                    # TorchScript로 컴파일 시도 (선택적)
                    pass
                except:
                    pass
            
            print(f"[Reranker] Model loaded successfully")
        except Exception as e:
            print(f"[Reranker] Error loading model: {e}")
            print(f"[Reranker] Falling back to default model")
            # 폴백: 기본 모델 사용
            self.model_name = self.MODEL_OPTIONS["fast"]
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def rerank(self, query, candidates, doc_store, top_k=10):
        """
        BM25 후보를 재순위화
        
        Args:
            query: 검색 쿼리
            candidates: [(doc_id, bm25_score), ...] BM25 결과
            doc_store: {doc_id: text} 문서 저장소
            top_k: 반환할 결과 수
        
        Returns: [(doc_id, rerank_score), ...]
        """
        if not candidates:
            return []
        
        # 문서 텍스트 가져오기
        docs = []
        for doc_id, _ in candidates:
            text = doc_store.get(doc_id, "")
            if text:
                # 텍스트 길이 제한 (모델에 따라 다름)
                max_length = 512 if "bge" in self.model_name.lower() else 1500
                docs.append((doc_id, text[:max_length]))
        
        if not docs:
            return []
        
        # 배치 처리
        scores = self._batch_predict(query, docs)
        
        # (doc_id, score) 쌍으로 정렬
        results = [(doc_id, score) for (doc_id, _), score in zip(docs, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _batch_predict(self, query, docs, batch_size=16):
        """배치 단위로 점수 예측 (CPU 최적화)"""
        all_scores = []
        
        # CPU에서는 더 작은 배치 크기 사용
        if self.device == "cpu":
            batch_size = min(batch_size, 8)
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            
            # BGE 모델은 다른 입력 형식 사용
            if "bge" in self.model_name.lower():
                # BGE: [query, doc] 형식
                pairs = [[query, text] for _, text in batch]
            else:
                # Cross-encoder: [query, doc] 형식
                pairs = [[query, text] for _, text in batch]
            
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 출력 형식 처리
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits.squeeze(-1)
                else:
                    logits = outputs.squeeze(-1)
                
                if logits.dim() == 0:
                    scores = [logits.item()]
                else:
                    scores = logits.cpu().tolist()
                
                all_scores.extend(scores)
        
        return all_scores


class LightweightReranker:
    """
    경량 리랭커 (임베딩 기반)
    Cross-Encoder보다 빠르지만 약간 낮은 성능
    """
    
    def __init__(self, device=None):
        from sentence_transformers import CrossEncoder
        
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 경량 모델 사용
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"[LightweightReranker] Loading: {model_name}")
        print(f"[LightweightReranker] Device: {self.device}")
        
        self.model = CrossEncoder(model_name, device=self.device)
    
    def rerank(self, query, candidates, doc_store, top_k=10):
        """임베딩 기반 재순위화"""
        if not candidates:
            return []
        
        docs = []
        for doc_id, _ in candidates:
            text = doc_store.get(doc_id, "")
            if text:
                docs.append((doc_id, text[:512]))
        
        if not docs:
            return []
        
        # 쿼리-문서 쌍 생성
        pairs = [[query, text] for _, text in docs]
        
        # 점수 예측
        scores = self.model.predict(pairs)
        
        # 정렬
        results = [(doc_id, float(score)) for (doc_id, _), score in zip(docs, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
