"""
데이터 다운로드 스크립트
wikir/en1k 데이터셋을 로컬에 저장
"""
import ir_datasets
import os
from tqdm import tqdm

DATA_DIR = "data"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 문서 저장 (training에서 한 번만)
    print("=" * 50)
    print("Downloading wikir/en1k dataset...")
    print("=" * 50)
    
    dataset = ir_datasets.load("wikir/en1k/training")
    
    # documents.tsv
    doc_path = os.path.join(DATA_DIR, "documents.tsv")
    if not os.path.exists(doc_path):
        print(f"\nSaving documents to {doc_path}")
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("doc_id\ttext\n")
            for doc in tqdm(dataset.docs_iter(), desc="Documents"):
                text = doc.text.replace('\t', ' ').replace('\n', ' ')
                f.write(f"{doc.doc_id}\t{text}\n")
    else:
        print(f"\n{doc_path} already exists, skipping...")
    
    # 각 split 처리
    for split in ["training", "validation", "test"]:
        print(f"\nProcessing {split}...")
        ds = ir_datasets.load(f"wikir/en1k/{split}")
        
        # queries
        qpath = os.path.join(DATA_DIR, f"queries_{split}.tsv")
        with open(qpath, 'w', encoding='utf-8') as f:
            f.write("query_id\ttext\n")
            for q in ds.queries_iter():
                text = q.text.replace('\t', ' ').replace('\n', ' ')
                f.write(f"{q.query_id}\t{text}\n")
        
        # qrels
        qrels_path = os.path.join(DATA_DIR, f"qrels_{split}.tsv")
        with open(qrels_path, 'w', encoding='utf-8') as f:
            f.write("query_id\tdoc_id\trelevance\n")
            for qrel in ds.qrels_iter():
                f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
        
        # TREC format
        trec_path = os.path.join(DATA_DIR, f"qrels_{split}.trec")
        with open(trec_path, 'w', encoding='utf-8') as f:
            for qrel in ds.qrels_iter():
                f.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)


if __name__ == "__main__":
    download()
