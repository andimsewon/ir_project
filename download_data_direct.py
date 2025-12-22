"""
데이터 다운로드 스크립트 (직접 다운로드 방식)
ir_datasets의 내부 압축 해제를 우회하여 ZIP 파일을 직접 다운로드하고 처리
"""
import os
import sys
import zipfile
import csv
import requests
from tqdm import tqdm
from pathlib import Path

DATA_DIR = "data"
ZIP_URL = "https://zenodo.org/record/3565761/files/wikIR1k.zip"
ZIP_FILENAME = "wikIR1k.zip"
EXTRACT_DIR = "wikIR1k_extracted"


def download_file(url, filename, chunk_size=8192):
    """파일을 직접 다운로드"""
    print(f"다운로드 중: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print(f"다운로드 완료: {filename}")


def extract_zip(zip_path, extract_to):
    """ZIP 파일 압축 해제"""
    print(f"압축 해제 중: {zip_path} -> {extract_to}")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file in tqdm(file_list, desc="압축 해제"):
            zip_ref.extract(file, extract_to)
    print("압축 해제 완료")


def process_csv_to_tsv(csv_path, tsv_path, header=None):
    """CSV 파일을 TSV로 변환"""
    with open(csv_path, 'r', encoding='utf-8') as csv_file, \
         open(tsv_path, 'w', encoding='utf-8') as tsv_file:
        
        csv.field_size_limit(min(2147483647, sys.maxsize))
        reader = csv.reader(csv_file)
        
        if header:
            tsv_file.write(header + '\n')
        
        for row in reader:
            cleaned_row = [cell.replace('\t', ' ').replace('\n', ' ') for cell in row]
            tsv_file.write('\t'.join(cleaned_row) + '\n')


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 50)
    print("wikir/en1k 데이터셋 직접 다운로드")
    print("=" * 50)
    
    zip_path = ZIP_FILENAME
    if not os.path.exists(zip_path):
        download_file(ZIP_URL, zip_path)
    else:
        print(f"{zip_path} 이미 존재합니다. 건너뜁니다.")
    
    if not os.path.exists(EXTRACT_DIR):
        extract_zip(zip_path, EXTRACT_DIR)
    else:
        print(f"{EXTRACT_DIR} 이미 존재합니다. 건너뜁니다.")
    
    extract_path = Path(EXTRACT_DIR)
    
    
    doc_csv = None
    for possible_path in [
        extract_path / "wikIR1k" / "documents.csv",
        extract_path / "documents.csv",
        extract_path / "training" / "documents.csv",
    ]:
        if possible_path.exists():
            doc_csv = possible_path
            break
    
    if doc_csv:
        doc_tsv = os.path.join(DATA_DIR, "documents.tsv")
        if os.path.exists(doc_tsv):
            print(f"\ndocuments.tsv 이미 존재합니다. 덮어쓰기 중...")
        else:
            print(f"\n문서 변환 중: {doc_csv}")
        process_csv_to_tsv(
            str(doc_csv),
            doc_tsv,
            header="doc_id\ttext"
        )
    else:
        print("\n경고: documents.csv를 찾을 수 없습니다.")
    
    for split in ["training", "validation", "test"]:
        print(f"\n처리 중: {split}")
        
        query_csv = None
        for possible_path in [
            extract_path / split / "queries.csv",
            extract_path / f"{split}_queries.csv",
            extract_path / "wikIR1k" / split / "queries.csv",
        ]:
            if possible_path.exists():
                query_csv = possible_path
                break
        
        if query_csv:
            qpath = os.path.join(DATA_DIR, f"queries_{split}.tsv")
            print(f"  Queries 변환: {query_csv}")
            process_csv_to_tsv(
                str(query_csv),
                qpath,
                header="query_id\ttext"
            )
        
        qrel_file = None
        for possible_path in [
            extract_path / split / "qrels",
            extract_path / "wikIR1k" / split / "qrels",
        ]:
            if possible_path.exists():
                qrel_file = possible_path
                break
        
        if qrel_file:
            qrels_path = os.path.join(DATA_DIR, f"qrels_{split}.tsv")
            trec_path = os.path.join(DATA_DIR, f"qrels_{split}.trec")
            
            print(f"  Qrels 처리: {qrel_file}")
            
            with open(qrel_file, 'r', encoding='utf-8') as f_in, \
                 open(qrels_path, 'w', encoding='utf-8') as f_tsv, \
                 open(trec_path, 'w', encoding='utf-8') as f_trec:
                f_tsv.write("query_id\tdoc_id\trelevance\n")
                for line in f_in:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 4:
                            query_id = parts[0]
                            doc_id = parts[2]
                            relevance = parts[3]
                            f_tsv.write(f"{query_id}\t{doc_id}\t{relevance}\n")
                            f_trec.write(f"{query_id} 0 {doc_id} {relevance}\n")
    
    print("\n" + "=" * 50)
    print("다운로드 및 변환 완료!")
    print("=" * 50)
    print(f"\n데이터 위치: {DATA_DIR}/")
    print(f"임시 파일 정리:")
    print(f"  - {zip_path} (ZIP 파일, 삭제 가능)")
    print(f"  - {EXTRACT_DIR}/ (압축 해제 폴더, 삭제 가능)")


if __name__ == "__main__":
    download()

