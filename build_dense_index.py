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


def build(
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str = None,
    ann_enabled: bool = True,
    ann_m: int = 32,
    ann_ef_construction: int = 200,
    ann_ef_search: int = 128,
):
    if not os.path.exists(INDEX_PATH):
        print(f"Error: {INDEX_PATH} not found")
        print("Run 'python build_index.py' first")
        return

    index = InvertedIndex()
    index.load(INDEX_PATH)

    retriever = DenseRetriever(
        model_name=model_name,
        device=device,
        max_length=max_length,
        ann_enabled=ann_enabled,
        ann_m=ann_m,
        ann_ef_construction=ann_ef_construction,
        ann_ef_search=ann_ef_search,
    )
    if ann_enabled and not retriever.ann_available:
        print("Warning: faiss not installed. ANN disabled; using exact search.")
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
    parser.add_argument("--no-ann", action="store_true", help="Disable FAISS ANN even if available")
    parser.add_argument("--ann-m", type=int, default=32, help="FAISS HNSW M parameter")
    parser.add_argument("--ann-efc", type=int, default=200, help="FAISS HNSW efConstruction")
    parser.add_argument("--ann-efsearch", type=int, default=128, help="FAISS HNSW efSearch")
    args = parser.parse_args()
    
    build(
        args.model,
        args.batch_size,
        args.max_length,
        args.device,
        ann_enabled=not args.no_ann,
        ann_m=args.ann_m,
        ann_ef_construction=args.ann_efc,
        ann_ef_search=args.ann_efsearch,
    )
