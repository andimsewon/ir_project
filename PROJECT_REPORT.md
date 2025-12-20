# 정보검색 기말 프로젝트 보고서

## 1. 프로젝트 개요

본 프로젝트는 정보검색의 핵심 구성 요소를 이해하고, 직접 구현한 검색 엔진을 통해 전체 IR 파이프라인을 구축하는 것을 목표로 합니다. wikir/en1k 데이터셋을 사용하여 검색 엔진을 구현하고, TREC Eval 지표를 활용한 정량적 성능 평가를 수행합니다.

### 프로젝트 목표
- 검색 엔진의 핵심 구조 이해
- 인덱싱, 검색, UI 구현, 성능 평가까지 전체 파이프라인 경험
- 다양한 랭킹 알고리즘의 성능 비교 및 분석

## 2. 시스템 구조

### 2.1 전체 아키텍처

```
┌─────────────┐
│  User Query │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Query Processor │ ◄── Query Expansion (Optional)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Inverted      │
│     Index       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Ranking Engine │ ◄── BM25 / TF-IDF / Hybrid
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   Reranker      │ ◄── Cross-Encoder (Optional)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Search Results  │
└─────────────────┘
```

### 2.2 모듈 구조

```
src/
├── indexer.py          # Inverted Index 구축
├── tokenizer.py        # 텍스트 토크나이저
├── ranker.py           # BM25 랭킹 알고리즘
├── tfidf_ranker.py    # TF-IDF 랭킹 알고리즘
├── searcher.py         # 통합 검색 엔진
├── query_expander.py   # 쿼리 확장 (선택 기능)
├── reranker.py         # Cross-Encoder 리랭킹
└── evaluator.py        # TREC Eval 평가 지표
```

## 3. 인덱싱 전략

### 3.1 Inverted Index 구현

**구조:**
```python
posting_list = {
    term: [(doc_id, term_frequency), ...],
    ...
}
doc_freq = {term: document_frequency, ...}
doc_len = {doc_id: document_length, ...}
doc_store = {doc_id: original_text, ...}
```

**구현 특징:**
- 모든 인덱싱 로직을 직접 구현
- Term Frequency (TF) 계산 및 저장
- Document Frequency (DF) 계산
- 문서 길이 정보 저장 (정규화용)
- 원본 문서 저장 (스니펫 생성용)

### 3.2 토크나이징

- 영어 텍스트 처리
- 소문자 변환
- Stopwords 제거
- 최소 길이 필터링 (2자 이상)

### 3.3 가중치 부여

**BM25 가중치:**
```
score(D,Q) = Σ IDF(q) × (tf × (k1+1)) / (tf + k1 × (1-b + b×|D|/avgdl))

IDF(q) = log((N - df + 0.5) / (df + 0.5) + 1)
```
- 파라미터: k1=1.5, b=0.75

**TF-IDF 가중치:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)

TF(t,d) = term_frequency / document_length
IDF(t) = log(N / DF(t))
```

## 4. 쿼리 처리

### 4.1 기본 쿼리 처리

1. **쿼리 토크나이징**: 사용자 입력을 토큰으로 분리
2. **Posting List 조회**: 각 쿼리 단어의 posting list 검색
3. **점수 계산**: BM25 또는 TF-IDF 점수 계산
4. **정렬**: 점수 내림차순 정렬

### 4.2 쿼리 확장 (선택 기능)

**동의어 확장:**
- 사전 기반 동의어 추가
- 예: "car" → "automobile", "vehicle"

**공출현 기반 확장:**
- 인덱스 통계를 활용한 관련어 확장
- 쿼리 단어와 함께 자주 나타나는 단어 탐색

### 4.3 랭킹 방법

1. **BM25**: 기본 랭킹 알고리즘
2. **TF-IDF**: 대안 랭킹 알고리즘
3. **Hybrid**: BM25와 TF-IDF 점수 가중 결합
   ```
   hybrid_score = α × BM25_score + (1-α) × TF-IDF_score
   ```

## 5. 사용자 인터페이스

### 5.1 웹 UI (Streamlit)

**주요 기능:**
- 쿼리 입력 필드
- 랭킹 방법 선택 (BM25 / TF-IDF / Hybrid)
- Reranker 옵션
- 쿼리 확장 옵션
- 검색 결과 표시:
  - 문서 ID
  - 점수
  - 스니펫 (쿼리 단어 하이라이팅)
  - 전체 문서 보기
- 페이지네이션
- 검색 히스토리
- 쿼리 분석 (단어 빈도, IDF)

### 5.2 UI 스크린샷

(실제 실행 시 스크린샷 추가)

## 6. 성능 평가 실험

### 6.1 평가 지표

다음 지표들을 직접 구현하여 평가:

- **MAP (Mean Average Precision)**: 전체 평균 정밀도
- **Precision@k**: 상위 k개 결과의 정밀도 (k=5, 10, 20)
- **Recall@k**: 상위 k개 결과의 재현율
- **nDCG@k**: 정규화된 할인 누적 이득

### 6.2 실험 설정

**데이터셋:**
- Training: 1,444 queries
- Validation: 100 queries (주요 평가용)
- Test: 100 queries (최종 평가용)

**비교 방법:**
1. BM25
2. TF-IDF
3. Hybrid (BM25 + TF-IDF)
4. BM25 + Cross-Encoder Reranker

### 6.3 실험 결과

#### Validation Set 결과

```
Metric       BM25     TF-IDF    Hybrid    Reranker
----------------------------------------------------
MAP         0.xxxx    0.xxxx    0.xxxx    0.xxxx
P@5         0.xxxx    0.xxxx    0.xxxx    0.xxxx
P@10        0.xxxx    0.xxxx    0.xxxx    0.xxxx
P@20        0.xxxx    0.xxxx    0.xxxx    0.xxxx
R@5         0.xxxx    0.xxxx    0.xxxx    0.xxxx
R@10        0.xxxx    0.xxxx    0.xxxx    0.xxxx
R@20        0.xxxx    0.xxxx    0.xxxx    0.xxxx
nDCG@5      0.xxxx    0.xxxx    0.xxxx    0.xxxx
nDCG@10     0.xxxx    0.xxxx    0.xxxx    0.xxxx
nDCG@20     0.xxxx    0.xxxx    0.xxxx    0.xxxx
```

(실제 실행 결과로 대체)

### 6.4 결과 분석

**주요 발견사항:**

1. **BM25 vs TF-IDF**: 
   - (실험 결과에 따른 분석)

2. **Hybrid 랭킹**:
   - (하이브리드 랭킹의 효과 분석)

3. **Reranker 효과**:
   - (Cross-Encoder 리랭킹의 성능 향상 분석)

4. **파라미터 영향**:
   - BM25 파라미터 (k1, b) 조정 실험
   - Hybrid 가중치 조정 실험

## 7. 구현 세부사항

### 7.1 직접 구현한 알고리즘

**인덱싱:**
- Inverted Index 구축 알고리즘
- TF 및 DF 계산
- 문서 길이 정규화

**랭킹:**
- BM25 스코어링 공식 구현
- TF-IDF 스코어링 공식 구현
- 하이브리드 랭킹 구현

**평가:**
- MAP 계산 알고리즘
- Precision@k 계산
- Recall@k 계산
- nDCG 계산 알고리즘

### 7.2 사용한 외부 라이브러리

**허용된 라이브러리:**
- `ir_datasets`: 데이터셋 다운로드
- `transformers`: HuggingFace 사전 학습 모델 (Cross-Encoder)
- `streamlit`: 웹 UI 프레임워크
- `torch`: 딥러닝 프레임워크 (Cross-Encoder용)

**금지된 라이브러리 (미사용):**
- Elasticsearch
- Lucene
- Solr
- Indri
- 기타 검색/인덱싱 라이브러리

## 8. 추가 기능 (창의적 요소)

### 8.1 하이브리드 랭킹

BM25와 TF-IDF의 장점을 결합하여 검색 성능 향상:
- BM25: 문서 길이 정규화 강점
- TF-IDF: 단순하고 직관적인 가중치
- 가중치 조절 가능 (사용자 설정)

### 8.2 쿼리 확장

- 동의어 확장: 검색 범위 확대
- 공출현 기반 확장: 관련어 자동 탐색

### 8.3 고급 UI 기능

- 검색 히스토리: 최근 검색 기록 관리
- 쿼리 분석: 단어별 통계 정보 제공
- 페이지네이션: 대량 결과 효율적 표시

## 9. 제약사항 준수

### 9.1 직접 구현

- ✅ 모든 인덱싱 로직 직접 구현
- ✅ 모든 랭킹 알고리즘 직접 구현
- ✅ 모든 평가 지표 직접 구현

### 9.2 외부 라이브러리 제한

- ✅ 검색/인덱싱 라이브러리 미사용
- ✅ HuggingFace 사전 학습 모델만 사용 (Reranking용)

## 10. 실행 방법

### 10.1 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 10.2 데이터 준비

```bash
# 데이터셋 다운로드
python download_data.py
```

### 10.3 인덱스 구축

```bash
# Inverted Index 구축
python build_index.py
```

### 10.4 성능 평가

```bash
# Validation set 평가
python run_eval.py --split validation

# Test set 평가
python run_eval.py --split test
```

### 10.5 웹 데모 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 접속

## 11. 결론

본 프로젝트를 통해 정보검색 시스템의 전체 파이프라인을 직접 구현하고, 다양한 랭킹 알고리즘의 성능을 비교 분석했습니다. 

**주요 성과:**
- 완전히 자체 구현한 검색 엔진
- BM25, TF-IDF, 하이브리드 랭킹 구현
- TREC Eval 표준 준수 평가 시스템
- 사용자 친화적인 웹 인터페이스

**향후 개선 방향:**
- Positional Index 구현 (구문 검색)
- n-gram 인덱싱 (부분 일치 검색)
- Relevance Feedback 구현
- 더 많은 쿼리 확장 기법

## 12. 참고문헌

- Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management.
- TREC Evaluation Guidelines: https://trec.nist.gov/

---

**프로젝트 기간:** 2025년 11월 - 12월  
**데이터셋:** wikir/en1k (ir_datasets)  
**프로그래밍 언어:** Python 3.x

