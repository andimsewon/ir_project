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
import contextlib
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

# Windows 파일 잠금 이슈를 피하기 위한 안전한 파일 완료 처리
import ir_datasets.util as _ir_util


@contextlib.contextmanager
def _finalized_file_safe(path, mode="wb", retries=8, base_delay=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, mode) as f:
        yield f
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(base_delay * (attempt + 1))
    # 최종 수단: 복사 후 정리 (이름 변경이 막혀도 결과 파일 보장)
    shutil.copyfile(tmp_path, path)
    for attempt in range(retries):
        try:
            os.remove(tmp_path)
            return
        except PermissionError:
            time.sleep(base_delay * (attempt + 1))


_ir_util.finialized_file = _finalized_file_safe

DATA_DIR = "data"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Windows에서 임시/캐시 디렉토리를 프로젝트 내부로 고정
    # 권한/잠금 문제를 줄이기 위해 고유 임시 디렉토리를 사용
    cache_dir = os.path.join(os.getcwd(), ".ir_datasets_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["IR_DATASETS_HOME"] = cache_dir

    base_tmp_root = os.path.join(os.getcwd(), "_tmp", "ir_datasets")
    os.makedirs(base_tmp_root, exist_ok=True)

    def _new_tmp_dir():
        tmp_dir = tempfile.mkdtemp(prefix="ir_datasets_tmp_", dir=base_tmp_root)
        os.environ["IR_DATASETS_TMP"] = tmp_dir
        return tmp_dir

    ir_datasets_tmp = _new_tmp_dir()
    
    # 문서 저장 (training에서 한 번만)
    print("=" * 50)
    print("Downloading wikir/en1k dataset...")
    print("=" * 50)
    print(f"임시 디렉토리: {ir_datasets_tmp}")
    
    # 재시도 로직을 포함한 데이터셋 로드 함수
    def load_dataset_with_retry(split_name, max_retries=8, retry_delay=7):
        """재시도 로직이 포함된 데이터셋 로드"""
        _new_tmp_dir()
        for attempt in range(max_retries):
            try:
                # Windows에서 파일이 완전히 해제될 때까지 대기
                if attempt > 0:
                    print(f"  재시도 {attempt}/{max_retries-1}... {retry_delay}초 대기 중...")
                    # 재시도 전에 임시 디렉토리 재설정
                    _new_tmp_dir()
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
                    print(f"  {base_tmp_root}")
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
        for retry in range(5):
            try:
                with open(qpath, 'w', encoding='utf-8') as f:
                    f.write("query_id\ttext\n")
                    for q in ds.queries_iter():
                        text = q.text.replace('\t', ' ').replace('\n', ' ')
                        f.write(f"{q.query_id}\t{text}\n")
                queries_success = True
                break
            except (PermissionError, OSError, Exception) as e:
                if retry < 4:
                    print(f"  Queries 처리 중 오류 발생 (재시도 {retry + 1}/5): {e}")
                    time.sleep(5 + retry * 2)
                    _new_tmp_dir()
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
        for retry in range(5):
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
            except (PermissionError, OSError, Exception) as e:
                if retry < 4:
                    print(f"  Qrels 처리 중 오류 발생 (재시도 {retry + 1}/5): {e}")
                    time.sleep(5 + retry * 2)
                    _new_tmp_dir()
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
