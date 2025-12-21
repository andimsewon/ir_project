"""
Build dense embedding index for retrieval.
"""
import argparse
import os

from src.dense_retriever import DenseRetriever
from src.indexer import InvertedIndex

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")
DENSE_INDEX_PATH = os.path.join(DATA_DIR, "dense_index.pt")


def build(model_name: str, batch_size: int, max_length: int, device: str = None):
    if not os.path.exists(INDEX_PATH):
        print(f"Error: {INDEX_PATH} not found")
        print("Run 'python build_index.py' first")
        return

    index = InvertedIndex()
    index.load(INDEX_PATH)

    retriever = DenseRetriever(model_name=model_name, device=device, max_length=max_length)
    retriever.build_index(index.doc_store, batch_size=batch_size, show_progress=True)
    retriever.save(DENSE_INDEX_PATH)

    print(f"Dense index saved to {DENSE_INDEX_PATH}")
    print(f"Documents indexed: {len(retriever.doc_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max tokens per document")
    parser.add_argument("--device", default=None, help="Device for encoding (e.g., cpu, cuda, xpu, dml)")
    args = parser.parse_args()
    
    build(args.model, args.batch_size, args.max_length, args.device)
