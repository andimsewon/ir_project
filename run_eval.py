"""
평가 실행 스크립트
BM25와 BM25+Reranker 성능 비교
"""
import os
import argparse
from src.indexer import InvertedIndex
from src.ranker import BM25Ranker
from src.tfidf_ranker import TFIDFRanker
from src.reranker import CrossEncoderReranker
from src.query_expander import QueryExpander
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
    
    print("Loading rankers...")
    bm25_ranker = BM25Ranker(index)
    tfidf_ranker = TFIDFRanker(index)
    reranker = CrossEncoderReranker()
    query_expander = QueryExpander(index)
    engine = SearchEngine(index, bm25_ranker, reranker, tfidf_ranker, query_expander)
    
    print("Loading evaluator...")
    evaluator = Evaluator(qrels_path, queries_path)
    
    # === BM25 평가 ===
    print("\n" + "-" * 60)
    print("Evaluating BM25...")
    print("-" * 60)
    
    run_bm25 = evaluator.generate_run(engine, method="bm25", use_reranker=False, top_k=100)
    evaluator.save_run_trec(
        run_bm25,
        os.path.join(RESULTS_DIR, f"bm25_{split}.txt"),
        "BM25"
    )
    results_bm25 = evaluator.evaluate(run_bm25)
    
    print("\nBM25 Results:")
    for metric, val in results_bm25.items():
        print(f"  {metric}: {val:.4f}")
    
    # === TF-IDF 평가 ===
    print("\n" + "-" * 60)
    print("Evaluating TF-IDF...")
    print("-" * 60)
    
    run_tfidf = evaluator.generate_run(engine, method="tfidf", use_reranker=False, top_k=100)
    evaluator.save_run_trec(
        run_tfidf,
        os.path.join(RESULTS_DIR, f"tfidf_{split}.txt"),
        "TF-IDF"
    )
    results_tfidf = evaluator.evaluate(run_tfidf)
    
    print("\nTF-IDF Results:")
    for metric, val in results_tfidf.items():
        print(f"  {metric}: {val:.4f}")
    
    # === Hybrid 평가 ===
    print("\n" + "-" * 60)
    print("Evaluating Hybrid (BM25 + TF-IDF)...")
    print("-" * 60)
    
    run_hybrid = evaluator.generate_run(engine, method="hybrid", use_reranker=False, top_k=100)
    evaluator.save_run_trec(
        run_hybrid,
        os.path.join(RESULTS_DIR, f"hybrid_{split}.txt"),
        "Hybrid"
    )
    results_hybrid = evaluator.evaluate(run_hybrid)
    
    print("\nHybrid Results:")
    for metric, val in results_hybrid.items():
        print(f"  {metric}: {val:.4f}")
    
    # === BM25 + Reranker 평가 ===
    print("\n" + "-" * 60)
    print("Evaluating BM25 + Reranker...")
    print("-" * 60)
    
    run_rerank = evaluator.generate_run(engine, method="bm25", use_reranker=True, top_k=100)
    evaluator.save_run_trec(
        run_rerank,
        os.path.join(RESULTS_DIR, f"rerank_{split}.txt"),
        "BM25+Reranker"
    )
    results_rerank = evaluator.evaluate(run_rerank)
    
    print("\nBM25 + Reranker Results:")
    for metric, val in results_rerank.items():
        print(f"  {metric}: {val:.4f}")
    
    # === 비교 ===
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"{'Metric':<12} {'BM25':>10} {'TF-IDF':>10} {'Hybrid':>10} {'Reranker':>10}")
    print("-" * 54)
    
    for metric in results_bm25:
        v_bm25 = results_bm25[metric]
        v_tfidf = results_tfidf[metric]
        v_hybrid = results_hybrid[metric]
        v_rerank = results_rerank[metric]
        print(f"{metric:<12} {v_bm25:>10.4f} {v_tfidf:>10.4f} {v_hybrid:>10.4f} {v_rerank:>10.4f}")
    
    # 결과 저장
    summary_path = os.path.join(RESULTS_DIR, f"summary_{split}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results - {split}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("BM25:\n")
        for m, v in results_bm25.items():
            f.write(f"  {m}: {v:.4f}\n")
        
        f.write("\nTF-IDF:\n")
        for m, v in results_tfidf.items():
            f.write(f"  {m}: {v:.4f}\n")
        
        f.write("\nHybrid (BM25 + TF-IDF):\n")
        for m, v in results_hybrid.items():
            f.write(f"  {m}: {v:.4f}\n")
        
        f.write("\nBM25 + Reranker:\n")
        for m, v in results_rerank.items():
            f.write(f"  {m}: {v:.4f}\n")
        
        f.write("\nComparison:\n")
        f.write(f"{'Metric':<12} {'BM25':>10} {'TF-IDF':>10} {'Hybrid':>10} {'Reranker':>10}\n")
        f.write("-" * 54 + "\n")
        for metric in results_bm25:
            v_bm25 = results_bm25[metric]
            v_tfidf = results_tfidf[metric]
            v_hybrid = results_hybrid[metric]
            v_rerank = results_rerank[metric]
            f.write(f"{metric:<12} {v_bm25:>10.4f} {v_tfidf:>10.4f} {v_hybrid:>10.4f} {v_rerank:>10.4f}\n")
    
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", 
                       choices=["training", "validation", "test"],
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    # Test set 사용 시 주의사항
    if args.split == "test":
        print("\n" + "!" * 60)
        print("WARNING: Using test set for evaluation.")
        print("Make sure you haven't tuned your system on test set!")
        print("!" * 60 + "\n")
    
    run_eval(args.split)
