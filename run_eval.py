"""
평가 실행 스크립트
BM25와 BM25+Reranker 성능 비교
"""
import os
import argparse
from src.indexer import InvertedIndex
from src.ranker import BM25Ranker
from src.reranker import CrossEncoderReranker
from src.searcher import SearchEngine
from src.evaluator import Evaluator

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
RESULTS_DIR = "results"


def run_eval(split="validation"):
    # 경로 설정
    qrels_path = os.path.join(DATA_DIR, f"qrels_{split}.tsv")
    queries_path = os.path.join(DATA_DIR, f"queries_{split}.tsv")
    
    # 파일 체크
    if not os.path.exists(INDEX_PATH):
        print("Error: Index not found. Run 'python build_index.py' first")
        return
    
    if not os.path.exists(qrels_path):
        print(f"Error: {qrels_path} not found. Run 'python download_data.py' first")
        return
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 컴포넌트 로드
    print("=" * 60)
    print(f"Evaluation on {split} set")
    print("=" * 60)
    
    print("\nLoading index...")
    index = InvertedIndex()
    index.load(INDEX_PATH)
    
    print("Loading reranker...")
    ranker = BM25Ranker(index)
    reranker = None
    reranker_available = True
    try:
        reranker = CrossEncoderReranker()
    except Exception as exc:
        reranker_available = False
        print(f"[Warning] Reranker disabled: {exc}")
    engine = SearchEngine(index, ranker, reranker)
    
    print("Loading evaluator...")
    evaluator = Evaluator(qrels_path, queries_path)
    
    # === BM25 평가 ===
    print("\n" + "-" * 60)
    print("Evaluating BM25...")
    print("-" * 60)
    
    run_bm25 = evaluator.generate_run(engine, use_reranker=False, top_k=100)
    evaluator.save_run_trec(
        run_bm25,
        os.path.join(RESULTS_DIR, f"bm25_{split}.txt"),
        "BM25"
    )
    results_bm25 = evaluator.evaluate(run_bm25)
    try:
        results_bm25_trec = evaluator.evaluate_pytrec(run_bm25, k=10)
    except RuntimeError as exc:
        print(f"[Error] {exc}")
        print("Install pytrec_eval and rerun evaluation.")
        return
    
    print("\nBM25 Results:")
    for metric, val in results_bm25.items():
        print(f"  {metric}: {val:.4f}")
    print("BM25 (pytrec_eval):")
    for metric, val in results_bm25_trec.items():
        print(f"  {metric}: {val:.4f}")
    
    # === BM25 + Reranker 평가 ===
    print("\n" + "-" * 60)
    print("Evaluating BM25 + Reranker...")
    print("-" * 60)
    
    results_rerank = None
    results_rerank_trec = None
    if reranker_available:
        run_rerank = evaluator.generate_run(engine, use_reranker=True, top_k=100)
        evaluator.save_run_trec(
            run_rerank,
            os.path.join(RESULTS_DIR, f"rerank_{split}.txt"),
            "BM25+Reranker"
        )
        results_rerank = evaluator.evaluate(run_rerank)
        results_rerank_trec = evaluator.evaluate_pytrec(run_rerank, k=10)
        
        print("\nBM25 + Reranker Results:")
        for metric, val in results_rerank.items():
            print(f"  {metric}: {val:.4f}")
        print("BM25 + Reranker (pytrec_eval):")
        for metric, val in results_rerank_trec.items():
            print(f"  {metric}: {val:.4f}")
    else:
        print("\nBM25 + Reranker Results: skipped (reranker unavailable)")
    
    # === 비교 ===
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"{'Metric':<12} {'BM25':>10} {'Reranker':>10} {'Diff':>10}")
    print("-" * 44)
    
    if results_rerank:
        for metric in results_bm25:
            v1 = results_bm25[metric]
            v2 = results_rerank[metric]
            diff = v2 - v1
            sign = "+" if diff > 0 else ""
            print(f"{metric:<12} {v1:>10.4f} {v2:>10.4f} {sign}{diff:>9.4f}")
    
    # 결과 저장
    summary_path = os.path.join(RESULTS_DIR, f"summary_{split}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results - {split}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("BM25:\n")
        for m, v in results_bm25.items():
            f.write(f"  {m}: {v:.4f}\n")
        f.write("\nBM25 (pytrec_eval):\n")
        for m, v in results_bm25_trec.items():
            f.write(f"  {m}: {v:.4f}\n")
        
        if results_rerank:
            f.write("\nBM25 + Reranker:\n")
            for m, v in results_rerank.items():
                f.write(f"  {m}: {v:.4f}\n")
            f.write("\nBM25 + Reranker (pytrec_eval):\n")
            for m, v in results_rerank_trec.items():
                f.write(f"  {m}: {v:.4f}\n")
        else:
            f.write("\nBM25 + Reranker:\n  skipped (reranker unavailable)\n")
        
        f.write("\nComparison:\n")
        f.write(f"{'Metric':<12} {'BM25':>10} {'Reranker':>10} {'Diff':>10}\n")
        f.write("-" * 44 + "\n")
        if results_rerank:
            for metric in results_bm25:
                v1 = results_bm25[metric]
                v2 = results_rerank[metric]
                diff = v2 - v1
                sign = "+" if diff > 0 else ""
                f.write(f"{metric:<12} {v1:>10.4f} {v2:>10.4f} {sign}{diff:>9.4f}\n")
        else:
            f.write("Reranker comparison skipped.\n")
    
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", 
                       choices=["training", "validation", "test"])
    args = parser.parse_args()
    
    run_eval(args.split)
