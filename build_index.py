"""
인덱스 빌드 스크립트
BM25 인덱스 구축 및 저장
"""
import os
from src.indexer import InvertedIndex

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "index.pkl")


def build():
    doc_path = os.path.join(DATA_DIR, "documents.tsv")
    
    if not os.path.exists(doc_path):
        print(f"Error: {doc_path} not found")
        print("Run 'python download_data.py' first")
        return
    
    print("=" * 50)
    print("Building BM25 Index")
    print("=" * 50)
    
    index = InvertedIndex()
    index.build_from_file(doc_path)
    index.save(INDEX_PATH)
    
    print("\n" + "=" * 50)
    print(f"Index saved to {INDEX_PATH}")
    print(f"Total documents: {index.total_docs}")
    print(f"Vocabulary size: {len(index.posting_list)}")
    print(f"Average doc length: {index.avg_doc_len:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    build()
