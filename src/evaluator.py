"""
평가 모듈
TREC Eval 지표 구현 (MAP, P@k, R@k, nDCG@k)
"""
import os
import math
from collections import defaultdict
from tqdm import tqdm


class Evaluator:
    """검색 성능 평가기"""
    
    def __init__(self, qrels_path, queries_path):
        """
        Args:
            qrels_path: relevance judgments 파일 경로
            queries_path: 쿼리 파일 경로
        """
        self.qrels = self._load_qrels(qrels_path)
        self.queries = self._load_queries(queries_path)
        print(f"[Evaluator] Loaded {len(self.queries)} queries, {len(self.qrels)} qrels")
    
    def _load_qrels(self, path):
        """qrels 파일 로드"""
        qrels = defaultdict(dict)
        with open(path, 'r', encoding='utf-8') as f:
            next(f)  # 헤더 스킵
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    qid, doc_id, rel = parts[0], parts[1], int(parts[2])
                    if rel > 0:
                        qrels[qid][doc_id] = rel
        return qrels
    
    def _load_queries(self, path):
        """쿼리 파일 로드"""
        queries = {}
        with open(path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    queries[parts[0]] = parts[1]
        return queries
    
    def generate_run(self, search_engine, method="bm25", use_reranker=False, 
                     use_query_expansion=False, top_k=100, hybrid_weight=0.5):
        """
        전체 쿼리에 대해 검색 실행
        
        Args:
            search_engine: SearchEngine 인스턴스
            method: "bm25", "tfidf", "hybrid"
            use_reranker: 리랭커 사용 여부
            use_query_expansion: 쿼리 확장 사용 여부
            top_k: 반환할 결과 수
            hybrid_weight: 하이브리드 랭킹 가중치
        
        Returns: {qid: [(doc_id, score), ...], ...}
        """
        run = {}
        
        for qid, query_text in tqdm(self.queries.items(), desc="Searching"):
            result = search_engine.search(
                query_text,
                method=method,
                top_k=top_k,
                use_reranker=use_reranker,
                use_query_expansion=use_query_expansion,
                hybrid_weight=hybrid_weight
            )
            run[qid] = [(r['doc_id'], r['score']) for r in result['results']]
        
        return run
    
    def evaluate(self, run, k_list=[5, 10, 20]):
        """
        검색 결과 평가
        
        Returns: {metric_name: value, ...}
        """
        metrics = {
            'MAP': [],
            **{f'P@{k}': [] for k in k_list},
            **{f'R@{k}': [] for k in k_list},
            **{f'nDCG@{k}': [] for k in k_list}
        }
        
        for qid in self.queries:
            if qid not in run:
                continue
            
            retrieved = [doc_id for doc_id, _ in run[qid]]
            relevant = self.qrels.get(qid, {})
            
            # MAP
            metrics['MAP'].append(self._ap(retrieved, relevant))
            
            # P@k, R@k, nDCG@k
            for k in k_list:
                metrics[f'P@{k}'].append(self._precision_at_k(retrieved, relevant, k))
                metrics[f'R@{k}'].append(self._recall_at_k(retrieved, relevant, k))
                metrics[f'nDCG@{k}'].append(self._ndcg_at_k(retrieved, relevant, k))
        
        # 평균 계산
        results = {}
        for name, values in metrics.items():
            results[name] = sum(values) / len(values) if values else 0.0
        
        return results
    
    def _precision_at_k(self, retrieved, relevant, k):
        """Precision@k"""
        if k == 0:
            return 0.0
        hits = sum(1 for d in retrieved[:k] if d in relevant)
        return hits / k
    
    def _recall_at_k(self, retrieved, relevant, k):
        """Recall@k"""
        if not relevant:
            return 0.0
        hits = sum(1 for d in retrieved[:k] if d in relevant)
        return hits / len(relevant)
    
    def _ap(self, retrieved, relevant):
        """Average Precision"""
        if not relevant:
            return 0.0
        
        hits = 0
        sum_prec = 0.0
        
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                hits += 1
                sum_prec += hits / (i + 1)
        
        return sum_prec / len(relevant)
    
    def _ndcg_at_k(self, retrieved, relevant, k):
        """nDCG@k"""
        if not relevant:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            rel = relevant.get(doc, 0)
            dcg += rel / math.log2(i + 2)
        
        # IDCG
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def save_run_trec(self, run, path, run_name="run"):
        """TREC 형식으로 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for qid, results in run.items():
                for rank, (doc_id, score) in enumerate(results, 1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")
        
        print(f"[Evaluator] Run saved to {path}")
