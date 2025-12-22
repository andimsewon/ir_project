# SAP (Search Anything Positively)

직접 구현한 IR(Information Retrieval) 검색엔진 프로젝트입니다. 역색인 기반의 BM25/TF-IDF/하이브리드 랭킹부터 쿼리 확장, 리랭커, Dense/SPLADE까지 실험할 수 있고 Streamlit UI로 데모가 가능합니다.

---

## 핵심 기능

- **Inverted Index 기반 검색**: 문서 토큰화, DF/TF, 평균 길이 등 직접 구축
- **랭킹 방법**: BM25, TF-IDF, Hybrid(BM25 + TF-IDF)
- **고급 기능**
  - 쿼리 확장: 동의어/공출현/임베딩 기반
  - Cross-Encoder 리랭커 (Top-K 재정렬)
- **Dense & SPLADE (옵션)**
  - Dense bi-encoder 인덱싱/검색
  - SPLADE sparse 인덱싱/검색
- **UI/CLI**
  - Streamlit 웹 UI
  - 터미널 검색 CLI
- **평가**
  - pytrec_eval 기반 MAP/P@k/nDCG 측정

---

## 데이터셋

- **wikir/en1k (ir_datasets)**
  - 문서: 369,712
  - 쿼리: training 1,444 / validation 100 / test 100
  - relevance: 0/1/2

---

## 빠른 시작 (Windows / macOS 공통)

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

### 3) 데이터 다운로드

```bash
python download_data.py
```

- 결과물: `data/` 하위에 `documents.tsv`, `queries_*.tsv`, `qrels_*.tsv`
- 다운로드가 막힐 경우 대안:

```bash
python download_data_direct.py
```

### 4) 인덱스 빌드 (BM25)

```bash
python build_index.py
```

- 결과물: `data/index.pkl`

### 5) Streamlit 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 내일 Mac 데모용 체크리스트 (권장)

### A안: Windows에서 만든 데이터/인덱스를 Mac으로 복사 (가장 빠름)

1) **Windows에서 아래 파일들을 통째로 복사**
   - `data/documents.tsv`
   - `data/queries_*.tsv`
   - `data/qrels_*.tsv`
   - `data/index.pkl`
   - (옵션) `data/dense_index.pt`, `data/splade_index.pt`
2) **Mac에서 프로젝트 폴더에 그대로 붙여넣기**
3) **Mac에서 가상환경 + 패키지 설치**
4) **데모 전 리허설 실행**

```bash
python check_data.py
streamlit run app.py
```

> 처음 실행 시 리랭커/임베딩 모델을 다운로드할 수 있습니다. 데모 전에 한번 실행해 캐시를 받아두면 안정적입니다.

### B안: Mac에서 바로 다운로드 + 인덱스 빌드

```bash
python download_data.py
python build_index.py
streamlit run app.py
```

> 네트워크와 시간 여유가 있을 때만 권장.

---

## Streamlit UI 사용법

- 상단 버튼으로 **BM25 / TF-IDF / 하이브리드 / 리랭커 / 쿼리 확장**을 선택
- 선택 상태가 `[ON]`으로 표시됨
- 검색 결과에 `방법`, `리랭커`, `쿼리 확장` 요약 캡션 표시
- 검색 중/완료 상태 표시

---

## 고급 검색 모드 (코드 레벨)

`src/searcher.py`의 `SearchEngine.search()`는 다음 `method`를 지원합니다:

- `bm25`
- `tfidf`
- `hybrid`
- `dense`
- `hybrid_dense`
- `splade`
- `hybrid_splade`

---

## Dense 인덱스 (옵션)

```bash
python build_dense_index.py --model BAAI/bge-base-en-v1.5 --batch-size 64 --max-length 512
```

- 결과물: `data/dense_index.pt`

---

## SPLADE 인덱스 (옵션)

```bash
python build_splade_index.py --model naver/splade-cocondenser-ensembledistil --batch-size 8 --max-length 256 --top-terms 128
```

- 결과물: `data/splade_index.pt`

---

## 평가 (TREC)

```bash
python run_eval.py --split validation
python run_eval.py --split test
```

- 결과: `results/` 폴더에 run 파일 및 요약 파일 생성

---

## CLI 검색 (터미널)

```bash
python cli_search.py --method bm25 --top-k 10
```

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
├─ build_dense_index.py     # Dense index 빌드
├─ build_splade_index.py    # SPLADE index 빌드
├─ run_eval.py              # 평가 실행
├─ check_data.py            # 데이터 확인
├─ requirements.txt
├─ src/
│  ├─ indexer.py
│  ├─ ranker.py
│  ├─ tfidf_ranker.py
│  ├─ searcher.py
│  ├─ query_expander.py
│  ├─ reranker.py
│  ├─ dense_retriever.py
│  ├─ splade_retriever.py
│  └─ evaluator.py
├─ data/                    # documents.tsv, queries_*.tsv, qrels_*.tsv, index.pkl
└─ results/                 # 평가 결과
```

---

## 트러블슈팅

- **Index not found**
  - `python download_data.py` → `python build_index.py`
- **모델 다운로드 지연**
  - 데모 전 한번 실행해 캐시 확보 (리랭커/임베딩 모델)
- **macOS에서 느림**
  - 기본은 CPU로 동작. 데모용으로는 충분
- **네트워크 불안**
  - A안(데이터/인덱스 복사)로 준비 권장

---

## 라이선스

교육용 프로젝트 목적.
