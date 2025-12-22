"""
Build SPLADE sparse index for retrieval.
"""
import argparse
import os

from src.indexer import InvertedIndex
from src.splade_retriever import SpladeRetriever

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
SPLADE_INDEX_PATH = os.path.join(DATA_DIR, "splade_index.pt")


def build(model_name: str, batch_size: int, max_length: int, top_terms: int, device: str = None):
    if not os.path.exists(INDEX_PATH):
        print(f"Error: {INDEX_PATH} not found")
        print("Run 'python build_index.py' first")
        return

    index = InvertedIndex()
    index.load(INDEX_PATH)

    retriever = SpladeRetriever(
        model_name=model_name,
        device=device,
        max_length=max_length,
        top_terms=top_terms,
    )
    retriever.build_index(index.doc_store, batch_size=batch_size, show_progress=True)
    retriever.save(SPLADE_INDEX_PATH)

    print(f"SPLADE index saved to {SPLADE_INDEX_PATH}")
    print(f"Documents indexed: {len(retriever.doc_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="naver/splade-cocondenser-ensembledistil", help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Max tokens per document")
    parser.add_argument("--top-terms", type=int, default=128, help="Top terms per document")
    parser.add_argument("--device", default=None, help="Device for encoding (e.g., cpu, cuda, xpu, dml)")
    args = parser.parse_args()

    build(args.model, args.batch_size, args.max_length, args.top_terms, args.device)
