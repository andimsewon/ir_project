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

_original_field_size_limit = csv.field_size_limit
def _safe_field_size_limit(new_limit):
    """안전한 csv.field_size_limit 래퍼"""
    max_safe_value = 2147483647
    safe_limit = min(new_limit, max_safe_value)
    try:
        return _original_field_size_limit(safe_limit)
    except OverflowError:
        return _original_field_size_limit(131072)  # 128KB

csv.field_size_limit = _safe_field_size_limit

import ir_datasets

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
    
    print("=" * 50)
    print("Downloading wikir/en1k dataset...")
    print("=" * 50)
    print(f"임시 디렉토리: {ir_datasets_tmp}")
    
    def load_dataset_with_retry(split_name, max_retries=8, retry_delay=7):
        """재시도 로직이 포함된 데이터셋 로드"""
        _new_tmp_dir()
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"  재시도 {attempt}/{max_retries-1}... {retry_delay}초 대기 중...")
                    _new_tmp_dir()
                    try:
                        tmp_ir_dir = os.path.join(tempfile.gettempdir(), "ir_datasets")
                        if os.path.exists(tmp_ir_dir):
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
    
    for split in ["training", "validation", "test"]:
        print(f"\nProcessing {split}...")
        
        ds = load_dataset_with_retry(split)
        
        time.sleep(3)
        
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
                    ds = load_dataset_with_retry(split, max_retries=3, retry_delay=3)
                else:
                    print(f"  Queries 처리 실패: {e}")
                    raise
        
        if not queries_success:
            raise Exception("Queries 저장 실패")
        
        qrels_path = os.path.join(DATA_DIR, f"qrels_{split}.tsv")
        trec_path = os.path.join(DATA_DIR, f"qrels_{split}.trec")
        
        qrels_success = False
        for retry in range(5):
            try:
                with open(qrels_path, 'w', encoding='utf-8') as f_tsv, \
                     open(trec_path, 'w', encoding='utf-8') as f_trec:
                    f_tsv.write("query_id\tdoc_id\trelevance\n")
                    for qrel in ds.qrels_iter():
                        f_tsv.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
                        f_trec.write(f"{qrel.query_id} 0 {qrel.doc_id} {qrel.relevance}\n")
                qrels_success = True
                break
            except (PermissionError, OSError, Exception) as e:
                if retry < 4:
                    print(f"  Qrels 처리 중 오류 발생 (재시도 {retry + 1}/5): {e}")
                    time.sleep(5 + retry * 2)
                    _new_tmp_dir()
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
