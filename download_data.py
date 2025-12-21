"""
데이터 다운로드 스크립트
wikir/en1k 데이터셋을 로컬에 저장
"""
import csv
import sys
import os
import time
import tempfile
import shutil
from tqdm import tqdm

# Windows에서 csv.field_size_limit 오버플로우 문제 해결
# ir_datasets 라이브러리가 내부에서 csv.field_size_limit을 호출할 때
# C long 타입의 최대값을 초과하지 않도록 모니키 패치 적용
_original_field_size_limit = csv.field_size_limit
def _safe_field_size_limit(new_limit):
    """안전한 csv.field_size_limit 래퍼"""
    # Windows C long 최대값 (2^31 - 1)
    max_safe_value = 2147483647
    safe_limit = min(new_limit, max_safe_value)
    try:
        return _original_field_size_limit(safe_limit)
    except OverflowError:
        # 더 작은 값으로 시도
        return _original_field_size_limit(131072)  # 128KB

csv.field_size_limit = _safe_field_size_limit

# 이제 ir_datasets를 import (패치 후)
import ir_datasets

DATA_DIR = "data"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Windows에서 임시 디렉토리 정리 및 환경 변수 설정
    # ir_datasets의 임시 디렉토리를 프로젝트 내부로 변경하여 권한 문제 완화
    ir_datasets_tmp = os.path.join(os.getcwd(), ".ir_datasets_tmp")
    os.makedirs(ir_datasets_tmp, exist_ok=True)
    
    # 환경 변수 설정 (ir_datasets가 사용할 수 있도록)
    original_tmpdir = os.environ.get('TMPDIR') or os.environ.get('TMP') or os.environ.get('TEMP')
    os.environ['IR_DATASETS_TMP'] = ir_datasets_tmp
    
    # 문서 저장 (training에서 한 번만)
    print("=" * 50)
    print("Downloading wikir/en1k dataset...")
    print("=" * 50)
    print(f"임시 디렉토리: {ir_datasets_tmp}")
    
    # 재시도 로직을 포함한 데이터셋 로드 함수
    def load_dataset_with_retry(split_name, max_retries=5, retry_delay=5):
        """재시도 로직이 포함된 데이터셋 로드"""
        for attempt in range(max_retries):
            try:
                # Windows에서 파일이 완전히 해제될 때까지 대기
                if attempt > 0:
                    print(f"  재시도 {attempt}/{max_retries-1}... {retry_delay}초 대기 중...")
                    # 재시도 전에 임시 파일 정리 시도
                    try:
                        tmp_ir_dir = os.path.join(tempfile.gettempdir(), "ir_datasets")
                        if os.path.exists(tmp_ir_dir):
                            # 잠긴 파일이 있을 수 있으므로 조심스럽게 처리
                            time.sleep(2)
                    except:
                        pass
                    time.sleep(retry_delay)
                return ir_datasets.load(f"wikir/en1k/{split_name}")
            except (PermissionError, OSError) as e:
                if attempt == max_retries - 1:
                    print(f"  최종 오류: {e}")
                    print(f"  팁: 안티바이러스가 파일을 스캔 중일 수 있습니다.")
                    print(f"  몇 분 후 다시 시도하거나, 임시 디렉토리를 수동으로 정리해보세요:")
                    print(f"  {os.path.join(tempfile.gettempdir(), 'ir_datasets')}")
                    raise
                print(f"  시도 {attempt + 1} 실패: {e}")
                continue
    
    dataset = load_dataset_with_retry("training")
    
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
        
        # 재시도 로직으로 데이터셋 로드
        ds = load_dataset_with_retry(split)
        
        # Windows에서 파일 처리 전 충분한 대기 (안티바이러스 스캔 완료 대기)
        time.sleep(3)
        
        # queries (재시도 로직 포함)
        qpath = os.path.join(DATA_DIR, f"queries_{split}.tsv")
        queries_success = False
        for retry in range(3):
            try:
        with open(qpath, 'w', encoding='utf-8') as f:
            f.write("query_id\ttext\n")
            for q in ds.queries_iter():
                text = q.text.replace('\t', ' ').replace('\n', ' ')
                f.write(f"{q.query_id}\t{text}\n")
                queries_success = True
                break
            except (PermissionError, OSError) as e:
                if retry < 2:
                    print(f"  Queries 처리 중 오류 발생 (재시도 {retry + 1}/3): {e}")
                    time.sleep(5)
                    # 데이터셋을 다시 로드
                    ds = load_dataset_with_retry(split, max_retries=3, retry_delay=3)
                else:
                    print(f"  Queries 처리 실패: {e}")
                    raise
        
        if not queries_success:
            raise Exception("Queries 저장 실패")
        
        # qrels (TSV와 TREC 형식을 동시에 작성, 재시도 로직 포함)
        qrels_path = os.path.join(DATA_DIR, f"qrels_{split}.tsv")
        trec_path = os.path.join(DATA_DIR, f"qrels_{split}.trec")
        
        qrels_success = False
        for retry in range(3):
            try:
                with open(qrels_path, 'w', encoding='utf-8') as f_tsv, \
                     open(trec_path, 'w', encoding='utf-8') as f_trec:
                    f_tsv.write("query_id\tdoc_id\trelevance\n")
            for qrel in ds.qrels_iter():
                        # TSV 형식
                        f_tsv.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
                        # TREC 형식
                        f_trec.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
                qrels_success = True
                break
            except (PermissionError, OSError) as e:
                if retry < 2:
                    print(f"  Qrels 처리 중 오류 발생 (재시도 {retry + 1}/3): {e}")
                    time.sleep(5)
                    # 데이터셋을 다시 로드
                    ds = load_dataset_with_retry(split, max_retries=3, retry_delay=3)
                else:
                    print(f"  Qrels 처리 실패: {e}")
                    raise
        
        if not qrels_success:
            raise Exception("Qrels 저장 실패")
    
    print("\n" + "=" * 50)
    print("Download complete!")
    print("=" * 50)


if __name__ == "__main__":
    download()
