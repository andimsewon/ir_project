"""
리랭커 모듈
Cross-Encoder 기반 재순위화
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CrossEncoderReranker:
    """
    Cross-Encoder 리랭커
    BM25 결과를 신경망 모델로 재순위화
    """
    
    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, device=None):
        # device 설정
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"[Reranker] Loading model: {self.MODEL_NAME}")
        print(f"[Reranker] Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
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
                docs.append((doc_id, text[:1500]))  # 길이 제한
        
        if not docs:
            return []
        
        # 배치 처리
        scores = self._batch_predict(query, docs)
        
        # (doc_id, score) 쌍으로 정렬
        results = [(doc_id, score) for (doc_id, _), score in zip(docs, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _batch_predict(self, query, docs, batch_size=32):
        """배치 단위로 점수 예측"""
        all_scores = []
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
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
                logits = self.model(**inputs).logits.squeeze(-1)
                
                if logits.dim() == 0:
                    scores = [logits.item()]
                else:
                    scores = logits.cpu().tolist()
                
                all_scores.extend(scores)
        
        return all_scores
