# 정보검색 기말 프로젝트

BM25 검색 엔진 구현 및 Cross-Encoder Reranking

## 프로젝트 구조

```
ir_final_project/
│
├── src/                    # 소스 코드 모듈
│   ├── __init__.py
│   ├── tokenizer.py        # 텍스트 토크나이저
│   ├── indexer.py          # Inverted Index 구축
│   ├── ranker.py           # BM25 스코어링
│   ├── reranker.py         # Cross-Encoder 리랭커
│   ├── searcher.py         # 통합 검색 엔진
│   └── evaluator.py        # TREC Eval 평가 지표
│
├── download_data.py        # 데이터셋 다운로드
├── build_index.py          # 인덱스 빌드
├── run_eval.py             # 성능 평가 실행
├── app.py                  # 웹 UI (Streamlit)
│
├── data/                   # 데이터 저장 (자동 생성)
├── results/                # 평가 결과 (자동 생성)
│
├── requirements.txt
└── README.md
```

## 환경 설정

### 1. Python 가상환경 생성

```bash
# 프로젝트 폴더에서 실행
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

### Step 1: 데이터 다운로드

```bash
python download_data.py
```

wikir/en1k 데이터셋을 다운로드하고 `data/` 폴더에 저장합니다.
- documents.tsv: 369,712개 문서
- queries_*.tsv: 쿼리 (train/validation/test)
- qrels_*.tsv: relevance judgments

### Step 2: 인덱스 빌드

```bash
python build_index.py
```

BM25 Inverted Index를 구축합니다.
- 출력: `data/index.pkl`

### Step 3: 성능 평가

```bash
# validation set으로 평가 (기본)
python run_eval.py

# test set으로 평가
python run_eval.py --split test
```

BM25와 BM25+Reranker 성능을 비교합니다.
- 결과: `results/` 폴더에 저장

### Step 4: 웹 데모 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

## 구현 내용

### 1. BM25 검색 엔진 (직접 구현)

**src/indexer.py**
- Inverted Index: `{term: [(doc_id, tf), ...]}`
- Document Store: 원본 문서 저장
- Document Length: 정규화용 문서 길이

**src/ranker.py**
- BM25 스코어링 공식:
  ```
  score(D,Q) = Σ IDF(q) × (tf × (k1+1)) / (tf + k1 × (1-b + b×|D|/avgdl))
  ```
- 파라미터: k1=1.5, b=0.75

### 2. Cross-Encoder Reranker

**src/reranker.py**
- 모델: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- BM25 top-100 후보를 재순위화
- HuggingFace transformers 사용

### 3. 평가 지표 (직접 구현)

**src/evaluator.py**
- MAP (Mean Average Precision)
- Precision@k (k=5, 10, 20)
- Recall@k
- nDCG@k

### 4. 웹 UI

**app.py**
- Streamlit 기반
- BM25 / BM25+Reranker 선택
- 쿼리 하이라이팅
- 스니펫 표시

## 평가 결과 예시

```
Metric       BM25     Reranker     Diff
--------------------------------------------
MAP         0.3xxx    0.4xxx     +0.1xxx
P@5         0.4xxx    0.5xxx     +0.1xxx
P@10        0.4xxx    0.5xxx     +0.1xxx
nDCG@10     0.4xxx    0.5xxx     +0.1xxx
```

## 데이터셋

- **wikir/en1k**: 위키피디아 기반 IR 데이터셋
- 문서: 369,712개
- 쿼리: train 1,444 / validation 100 / test 100
- Relevance levels: 0 (not relevant), 1 (relevant), 2 (highly relevant)

## 참고

- 검색 라이브러리 (elasticsearch, lucene 등) 사용하지 않음
- HuggingFace 사전학습 모델만 사용 (과제 요구사항)
