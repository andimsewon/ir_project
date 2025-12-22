"""
수동으로 다운로드한 파일 처리 스크립트
직접 다운로드하고 압축 해제한 wikIR1k 폴더를 처리합니다.
"""
import os
import sys
import csv
from pathlib import Path

DATA_DIR = "data"


def process_csv_to_tsv(csv_path, tsv_path, header=None):
    """CSV 파일을 TSV로 변환"""
    print(f"  변환 중: {csv_path} -> {tsv_path}")
    with open(csv_path, 'r', encoding='utf-8') as csv_file, \
         open(tsv_path, 'w', encoding='utf-8') as tsv_file:
        
        # CSV 리더 설정 (큰 필드 크기 허용)
        csv.field_size_limit(min(2147483647, sys.maxsize))
        reader = csv.reader(csv_file)
        
        # 헤더 작성
        if header:
            tsv_file.write(header + '\n')
        
        # 데이터 변환
        for row in reader:
            # 탭과 줄바꿈을 공백으로 대체
            cleaned_row = [cell.replace('\t', ' ').replace('\n', ' ') for cell in row]
            tsv_file.write('\t'.join(cleaned_row) + '\n')


def process_qrels(qrel_path, qrels_tsv_path, qrels_trec_path):
    """Qrels 파일 처리 (TREC 형식)"""
    print(f"  처리 중: {qrel_path}")
    with open(qrel_path, 'r', encoding='utf-8') as f_in, \
         open(qrels_tsv_path, 'w', encoding='utf-8') as f_tsv, \
         open(qrels_trec_path, 'w', encoding='utf-8') as f_trec:
        f_tsv.write("query_id\tdoc_id\trelevance\n")
        for line in f_in:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 4:
                    query_id = parts[0]
                    doc_id = parts[2]
                    relevance = parts[3]
                    # TSV 형식
                    f_tsv.write(f"{query_id}\t{doc_id}\t{relevance}\n")
                    # TREC 형식
                    f_trec.write(f"{query_id} 0 {doc_id} {relevance}\n")


def process_manual_download(source_dir):
    """
    수동으로 다운로드한 wikIR1k 폴더를 처리
    
    Args:
        source_dir: wikIR1k 폴더가 있는 경로 (예: "wikIR1k_extracted/wikIR1k" 또는 직접 압축 해제한 폴더)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"오류: {source_dir} 폴더를 찾을 수 없습니다.")
        print("다운로드한 ZIP 파일을 압축 해제한 폴더 경로를 입력하세요.")
        return
    
    print("=" * 50)
    print("수동 다운로드 파일 처리")
    print("=" * 50)
    print(f"소스 디렉토리: {source_path}")
    
    # documents.tsv
    doc_csv = source_path / "documents.csv"
    if doc_csv.exists():
        doc_tsv = os.path.join(DATA_DIR, "documents.tsv")
        print(f"\n문서 처리:")
        process_csv_to_tsv(str(doc_csv), doc_tsv, header="doc_id\ttext")
    else:
        print(f"\n경고: {doc_csv}를 찾을 수 없습니다.")
    
    # 각 split 처리
    for split in ["training", "validation", "test"]:
        split_path = source_path / split
        if not split_path.exists():
            print(f"\n경고: {split_path}를 찾을 수 없습니다. 건너뜁니다.")
            continue
        
        print(f"\n처리 중: {split}")
        
        # queries
        query_csv = split_path / "queries.csv"
        if query_csv.exists():
            qpath = os.path.join(DATA_DIR, f"queries_{split}.tsv")
            process_csv_to_tsv(str(query_csv), qpath, header="query_id\ttext")
        else:
            print(f"  경고: {query_csv}를 찾을 수 없습니다.")
        
        # qrels
        qrel_file = split_path / "qrels"
        if qrel_file.exists():
            qrels_tsv = os.path.join(DATA_DIR, f"qrels_{split}.tsv")
            qrels_trec = os.path.join(DATA_DIR, f"qrels_{split}.trec")
            process_qrels(str(qrel_file), qrels_tsv, qrels_trec)
        else:
            print(f"  경고: {qrel_file}를 찾을 수 없습니다.")
    
    print("\n" + "=" * 50)
    print("처리 완료!")
    print("=" * 50)
    print(f"\n데이터 위치: {DATA_DIR}/")


if __name__ == "__main__":
    # 명령줄 인자로 경로를 받거나, 기본 경로 사용
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
    else:
        # 기본 경로들 시도
        possible_paths = [
            "wikIR1k_extracted/wikIR1k",
            "wikIR1k",
            ".",
        ]
        source_dir = None
        for path in possible_paths:
            if Path(path).exists():
                # documents.csv가 있는지 확인
                if (Path(path) / "documents.csv").exists() or \
                   any((Path(path) / split / "queries.csv").exists() for split in ["training", "validation", "test"]):
                    source_dir = path
                    break
        
        if not source_dir:
            print("사용법: python process_manual_download.py <wikIR1k_폴더_경로>")
            print("\n또는 wikIR1k 폴더를 현재 디렉토리에 두고 실행하세요.")
            print("\n예시:")
            print("  python process_manual_download.py wikIR1k_extracted/wikIR1k")
            print("  python process_manual_download.py C:/Downloads/wikIR1k")
            sys.exit(1)
    
    process_manual_download(source_dir)




