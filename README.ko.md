# SAP (Search Anything Positively)

직접 구현한 IR(Information Retrieval) 검색 엔진 프로젝트입니다. 역색인 기반 BM25/TF-IDF 랭킹부터 쿼리 확장, 리랭커, Dense/SPLADE, ANN(FAISS)까지 실험할 수 있고, Streamlit UI로 데모가 가능합니다.

---

## 목차

- 프로젝트 개요
- 시스템 아키텍처
- 주요 기능
- 실행 방법 (Windows / macOS)
- 평가 (TREC)
- 프로젝트 구조
- 제약사항 준수
- 트러블슈팅

---

## 프로젝트 개요

### 목표
- 검색 엔진의 핵심 구조 이해
- 인덱싱, 검색, 랭킹, UI, 평가까지 전체 파이프라인 구현
- 다양한 랭킹 알고리즘 성능 비교 및 분석

### 특징
- 완전 자체 구현: 인덱싱/랭킹 로직 직접 구현
- 다양한 랭킹 방법: BM25, TF-IDF
- 고급 기능: 쿼리 확장, Cross-Encoder 리랭커
- 평가: pytrec_eval 기반 MAP/P@k/nDCG
- UI: Streamlit 웹 UI + 터미널 CLI

### 데이터셋
- wikir/en1k (ir_datasets)
- 문서: 369,712
- 쿼리: training 1,444 / validation 100 / test 100
- relevance: 0 (not relevant), 1 (relevant), 2 (highly relevant)

---

## 시스템 아키텍처

### 전체 파이프라인

사용자 쿼리 입력
→ Query Processor (토크나이징, 쿼리 확장 선택)
→ Retrieval/Ranking (BM25/TF-IDF/Dense/SPLADE)
→ Reranker (선택, Cross-Encoder)
→ 결과 출력 (스니펫/하이라이트, 페이지네이션)

### 핵심 구성 요소
- Inverted Index: `src/indexer.py`
- Rankers: `src/ranker.py` (BM25), `src/tfidf_ranker.py`
- Search Engine: `src/searcher.py`
- Query Expander: `src/query_expander.py`
- Reranker: `src/reranker.py`
- Dense/SPLADE: `src/dense_retriever.py`, `src/splade_retriever.py`
- ANN(FAISS): `src/dense_retriever.py`
- UI: `app.py` (Streamlit), `cli_search.py` (CLI)

---

## 주요 기능

### 필수 기능
- Inverted Index 구축 (posting list, DF, doc length, doc store)
- BM25 랭킹 (k1=1.5, b=0.75)
- TF-IDF 랭킹
- Streamlit 웹 UI (SPLADE 고정 + 리랭커/쿼리 확장 토글)

### 추가 기능
- 쿼리 확장 (동의어/공출현/임베딩 기반)
- Cross-Encoder 리랭킹 (`BAAI/bge-reranker-base` 기본)
- 하이라이팅, 페이지네이션, 토글 옵션 표시
- Dense retrieval + ANN (옵션, `BAAI/bge-base-en-v1.5`, FAISS HNSW)
- SPLADE (필수, `naver/splade-cocondenser-ensembledistil`)

---

## 실행 방법 (Windows / macOS)

### 1) 가상환경

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2) 패키지 설치

```bash
pip install -r requirements.txt
```

ANN(FAISS) 사용 시:

```bash
pip install faiss-cpu
```

### 3) 데이터 다운로드

```bash
python download_data.py
```

대안 (직접 다운로드):

```bash
python download_data_direct.py
```

### 4) BM25 인덱스 빌드

```bash
python build_index.py
```

### 4-1) SPLADE 인덱스 빌드 (필수, GPU 권장)

Intel Arc A770 (DirectML) 사용:

```bash
pip install torch-directml
python build_splade_index.py --device dml
```

CPU 사용:

```bash
python build_splade_index.py --device cpu
```

SPLADE 인덱스를 생성해야 검색이 동작합니다.
Streamlit/CLI는 `torch-directml`이 설치되어 있으면 자동으로 DirectML을 사용합니다.

### 4-2) Dense + ANN 인덱스 빌드 (선택)

FAISS가 설치되어 있으면 ANN 인덱스를 자동 생성하며, 평가 시 ANN 검색을 사용합니다.
ANN 검색은 CPU에서 동작하고, 임베딩 인코딩은 `--device dml`로 Intel Arc GPU를 사용합니다.
현재 Streamlit/CLI는 SPLADE 전용으로 동작합니다. Dense ANN은 별도 실험용 인덱스입니다.

```bash
python build_dense_index.py --device dml
```

GPU 메모리 부족 오류가 나면 배치/길이를 줄이세요:

```bash
python build_dense_index.py --device dml --batch-size 8 --max-length 256
```

### 5) Streamlit UI 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

### 5-1) 웹 UI 사용 방법 (SPLADE 고정)

1) 검색어 입력 후 Enter 또는 `검색` 버튼 클릭  
2) 결과 카드에서 제목/스니펫/점수 확인  
3) 페이지네이션으로 결과 이동  
4) Reranker / Query Expansion 토글로 옵션 변경  

### 6) CLI 실행 (선택)

```bash
python cli_search.py --top-k 10
```

---

## Mac 데모 빠른 준비 (권장)

### A안: Windows에서 만든 데이터/인덱스 복사

1) Windows에서 아래 파일 복사
   - `data/documents.tsv`
   - `data/queries_*.tsv`
   - `data/qrels_*.tsv`
   - `data/index.pkl`
   - (옵션) `data/dense_index.pt`, `data/splade_index.pt`
2) Mac에 같은 경로로 복사
3) Mac에서 venv + `pip install -r requirements.txt`
4) 리허설:

```bash
python check_data.py
streamlit run app.py
```

### B안: Mac에서 직접 다운로드 + 빌드

```bash
python download_data.py
python build_index.py
streamlit run app.py
```

---

## 평가 (TREC)

### 실행

```bash
python run_eval.py --split validation
python run_eval.py --split test
```

SPLADE 평가:

```bash
python run_eval.py --split validation --method splade
python run_eval.py --split test --method splade
```

Dense ANN 평가:

```bash
python run_eval.py --split validation --method dense_ann
python run_eval.py --split test --method dense_ann
```

BM25 vs Reranker 비교(기본):

```bash
python run_eval.py --split validation
python run_eval.py --split test
```

### 결과 위치
- `results/summary_validation.txt`
- `results/summary_test.txt`
- `results/bm25_*.txt`, `results/tfidf_*.txt`, `results/hybrid_*.txt`, `results/rerank_*.txt`
- `results/summary_splade_*.txt`, `results/splade_*.txt`
- `results/summary_dense_ann_*.txt`, `results/dense_ann_*.txt`

### 측정 지표
- MAP, Precision@k, Recall@k, nDCG

### 결과 요약 (현재 results 기준)

Validation (BM25 vs BM25+Reranker)

| Metric | BM25 | BM25 + Reranker |
| --- | --- | --- |
| MAP | 0.1495 | 0.1372 |
| P@10 | 0.2080 | 0.1940 |
| nDCG@10 | 0.3078 | 0.3178 |

Test (BM25 vs BM25+Reranker)

| Metric | BM25 | BM25 + Reranker |
| --- | --- | --- |
| MAP | 0.1754 | 0.1535 |
| P@10 | 0.2120 | 0.1990 |
| nDCG@10 | 0.3584 | 0.3405 |

참고: 이번 결과에서는 Reranker가 Validation에서 nDCG@10은 상승했지만 MAP/P@10은 하락했고, Test에서는 전체적으로 하락했습니다.

---

## 프로젝트 구조

```
ir_project/
├─ app.py                   # Streamlit UI
├─ cli_search.py            # 터미널 UI
├─ download_data.py         # ir_datasets 기반 다운로드
├─ download_data_direct.py  # 직접 다운로드 대안
├─ process_manual_download.py
├─ build_index.py           # BM25 index 빌드
├─ build_dense_index.py     # Dense + ANN index 빌드
├─ build_splade_index.py    # SPLADE index 빌드
├─ run_eval.py              # 평가 실행
├─ check_data.py            # 데이터 확인
├─ requirements.txt
├─ src/
│  ├─ indexer.py
│  ├─ tokenizer.py
│  ├─ ranker.py
│  ├─ tfidf_ranker.py
│  ├─ searcher.py
│  ├─ query_expander.py
│  ├─ reranker.py
│  ├─ dense_retriever.py
│  ├─ splade_retriever.py
│  └─ evaluator.py
├─ data/                    # documents.tsv, queries_*.tsv, qrels_*.tsv, index.pkl, splade_index.pt, dense_index.pt, dense_index.faiss
└─ results/                 # 평가 결과
```

---

## 제약사항 준수

- 외부 검색 엔진 라이브러리 미사용 (Elasticsearch/Lucene/Solr/Indri)
- HuggingFace 사전 학습 모델만 사용
- 인덱싱/랭킹 로직은 직접 구현

---

## 트러블슈팅

- Index not found
  - `python download_data.py` → `python build_index.py` → `python build_splade_index.py`
- 모델 다운로드 지연
  - 데모 전 한 번 실행하여 캐시 확보
- macOS에서 느림
  - CPU로 동작하므로 데모용으로 충분
- 네트워크 불안
  - A안(데이터/인덱스 복사) 권장
- faiss-cpu 설치 실패 (Windows)
  - 파이썬/환경에 따라 설치가 실패할 수 있습니다. 이 경우 Dense ANN은 생략하고 `--no-ann`으로 빌드하세요.
