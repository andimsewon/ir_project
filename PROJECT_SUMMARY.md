# 정보검색 기말 프로젝트 - 요구사항 체크리스트

## ✅ 필수 요구사항 충족 여부

### 1. 검색 엔진 구현

#### 인덱싱 (Indexing)
- ✅ **Inverted Index 구축**: `src/indexer.py`에서 직접 구현
  - 구조: `{term: [(doc_id, tf), ...]}`
  - Document Store: 원본 문서 저장
  - Document Length: 정규화용 문서 길이 저장
  
- ✅ **TF-IDF 가중치 부여**: `src/tfidf_ranker.py`에서 직접 구현
  - TF 계산: 정규화된 빈도
  - IDF 계산: log(N / DF(t))
  
- ✅ **BM25 가중치 부여**: `src/ranker.py`에서 직접 구현
  - 파라미터: k1=1.5, b=0.75
  - 문서 길이 정규화 포함

#### 검색 (Query Processing)
- ✅ **사용자 질의 처리**: `src/searcher.py`에서 구현
- ✅ **BM25 랭킹 알고리즘**: 직접 구현
- ✅ **TF-IDF 랭킹 알고리즘**: 직접 구현
- ✅ **하이브리드 랭킹**: BM25와 TF-IDF 가중 결합 (창의적 기능)

#### UI 구성
- ✅ **웹 UI**: Streamlit 기반 (`app.py`)
- ✅ **질의 입력**: 텍스트 입력 필드
- ✅ **검색 결과 표시**: 문서 제목, 스니펫, 점수 표시
- ✅ **하이라이팅**: 쿼리 단어 하이라이팅 기능
- ✅ **페이지네이션**: 결과 페이지 분할 (창의적 기능)

### 2. TREC Eval 기반 성능 평가

- ✅ **MAP (Mean Average Precision)**: `src/evaluator.py`에서 직접 구현
- ✅ **Precision@k**: k=5, 10, 20 지원
- ✅ **Recall**: Recall@k 구현
- ✅ **nDCG**: nDCG@k 구현
- ✅ **TREC 형식 저장**: `run_eval.py`에서 결과 저장
- ✅ **성능 비교**: BM25, TF-IDF, Hybrid, Reranker 비교 리포트 생성

### 3. 제약사항 준수

- ✅ **외부 검색 라이브러리 미사용**: 
  - Elasticsearch, Lucene, Solr 등 사용하지 않음
  - 모든 인덱싱 및 검색 알고리즘 직접 구현
  
- ✅ **HuggingFace 모델 사용 허용**:
  - Cross-Encoder 리랭커에서 `cross-encoder/ms-marco-MiniLM-L-6-v2` 사용
  - 사전 학습된 모델만 사용 (과제 요구사항 준수)

## 🎨 추가 기능 (창의적 요소)

### 1. 하이브리드 랭킹
- BM25와 TF-IDF 점수를 가중 결합하여 검색 성능 향상
- 사용자가 가중치 조절 가능 (0.0 ~ 1.0)

### 2. 쿼리 확장 (Query Expansion)
- **동의어 확장**: 사전 기반 동의어 추가
- **공출현 기반 확장**: 인덱스 통계를 활용한 관련어 확장
- 쿼리 품질 향상으로 검색 성능 개선

### 3. Cross-Encoder 리랭킹
- BM25/TF-IDF top-100 후보를 신경망 모델로 재순위화
- 검색 정확도 향상

### 4. 고급 UI 기능
- **검색 히스토리**: 최근 검색 기록 저장 및 재검색
- **쿼리 분석**: 쿼리 단어의 Document Frequency, IDF 표시
- **확장 쿼리 표시**: 쿼리 확장 사용 시 원본/확장 쿼리 비교
- **페이지네이션**: 대량 결과를 페이지별로 분할 표시

### 5. 성능 평가 개선
- BM25, TF-IDF, Hybrid, Reranker 성능 비교 리포트
- TREC 형식 결과 파일 생성
- 상세한 성능 지표 비교 테이블

## 📁 프로젝트 구조

```
ir_project/
├── src/
│   ├── indexer.py          # Inverted Index 구축
│   ├── tokenizer.py        # 텍스트 토크나이저
│   ├── ranker.py           # BM25 랭킹
│   ├── tfidf_ranker.py    # TF-IDF 랭킹 (신규)
│   ├── searcher.py         # 통합 검색 엔진
│   ├── query_expander.py   # 쿼리 확장 (신규)
│   ├── reranker.py         # Cross-Encoder 리랭킹
│   └── evaluator.py        # TREC Eval 평가
├── app.py                  # Streamlit 웹 UI
├── build_index.py          # 인덱스 빌드 스크립트
├── run_eval.py             # 성능 평가 스크립트
├── download_data.py        # 데이터셋 다운로드
├── requirements.txt
└── README.md
```

## 🚀 실행 방법

1. **데이터 다운로드**
   ```bash
   python download_data.py
   ```

2. **인덱스 빌드**
   ```bash
   python build_index.py
   ```

3. **성능 평가**
   ```bash
   python run_eval.py --split validation
   ```

4. **웹 데모 실행**
   ```bash
   streamlit run app.py
   ```

## 📊 평가 지표

프로젝트는 다음 지표로 평가됩니다:
- MAP (Mean Average Precision)
- Precision@5, Precision@10, Precision@20
- Recall@5, Recall@10, Recall@20
- nDCG@5, nDCG@10, nDCG@20

## ✨ 주요 특징

1. **완전 자체 구현**: 모든 검색 알고리즘 직접 구현
2. **다양한 랭킹 방법**: BM25, TF-IDF, 하이브리드 지원
3. **고급 기능**: 쿼리 확장, 리랭킹, 성능 분석
4. **사용자 친화적 UI**: 직관적인 웹 인터페이스
5. **체계적인 평가**: TREC Eval 표준 준수

## 📝 참고사항

- 모든 검색 관련 알고리즘은 직접 구현
- HuggingFace 사전 학습 모델만 사용 (과제 요구사항 준수)
- 외부 검색 라이브러리 사용하지 않음
- TREC Eval 형식으로 결과 저장 및 비교

