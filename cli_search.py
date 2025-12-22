"""
Simple terminal UI for the search engine.
"""
import argparse
import os

from src.indexer import InvertedIndex
from src.ranker import BM25Ranker
from src.tfidf_ranker import TFIDFRanker
from src.searcher import SearchEngine


def load_engine(index_path):
    if not os.path.exists(index_path):
        print(f"Error: {index_path} not found. Run 'python download_data.py' and 'python build_index.py' first.")
        return None

    index = InvertedIndex()
    index.load(index_path)

    bm25_ranker = BM25Ranker(index)
    tfidf_ranker = TFIDFRanker(index)

    return SearchEngine(index, bm25_ranker, tfidf_ranker=tfidf_ranker)


def main():
    parser = argparse.ArgumentParser(description="Terminal search UI")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to show")
    parser.add_argument(
        "--method",
        default="bm25",
        choices=["bm25", "tfidf", "hybrid"],
        help="Ranking method",
    )
    args = parser.parse_args()

    engine = load_engine(os.path.join("data", "index.pkl"))
    if engine is None:
        return

    print("Enter a query (empty line or 'exit' to quit).")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except EOFError:
            break

        if not query or query.lower() in {"exit", "quit"}:
            break

        result = engine.search(query, top_k=args.top_k, method=args.method)
        print(f"\nTop {len(result['results'])} results ({result['method']}):")
        for item in result["results"]:
            doc_id = item["doc_id"]
            score = item["score"]
            snippet = item.get("snippet", "")
            print(f"- {doc_id} | score={score:.4f}")
            if snippet:
                print(f"  {snippet}")


if __name__ == "__main__":
    main()
