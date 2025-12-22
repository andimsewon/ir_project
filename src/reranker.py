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
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
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
            
            if self.device == "cpu":
                pass
            
            print(f"[Reranker] Model loaded successfully")
        except Exception as e:
            print(f"[Reranker] Error loading model: {e}")
            print(f"[Reranker] Falling back to default model")
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
        
        docs = []
        for doc_id, _ in candidates:
            text = doc_store.get(doc_id, "")
            if text:
                max_length = 512 if "bge" in self.model_name.lower() else 1500
                docs.append((doc_id, text[:max_length]))
        
        if not docs:
            return []
        
        scores = self._batch_predict(query, docs)
        
        results = [(doc_id, score) for (doc_id, _), score in zip(docs, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _batch_predict(self, query, docs, batch_size=16):
        """배치 단위로 점수 예측 (CPU 최적화)"""
        all_scores = []
        
        if self.device == "cpu":
            batch_size = min(batch_size, 8)
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            
            if "bge" in self.model_name.lower():
                pairs = [[query, text] for _, text in batch]
            else:
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
        
        pairs = [[query, text] for _, text in docs]
        
        scores = self.model.predict(pairs)
        
        results = [(doc_id, float(score)) for (doc_id, _), score in zip(docs, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
