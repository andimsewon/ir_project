# 제출 가이드

## 제출 전 체크리스트

### 1. 필수 파일 확인

#### 포함해야 할 파일
- ✅ `PROJECT_REPORT.md` - 프로젝트 보고서
- ✅ `README.md` - 사용 설명서
- ✅ `requirements.txt` - 패키지 목록
- ✅ `src/` - 모든 소스 코드
- ✅ `download_data.py` - 데이터 다운로드 스크립트
- ✅ `build_index.py` - 인덱스 빌드 스크립트
- ✅ `run_eval.py` - 평가 실행 스크립트
- ✅ `app.py` - 웹 UI
- ✅ `results/` - TREC Eval 결과 파일

#### 제외해야 할 파일/폴더
- ❌ `data/documents.tsv` - 원본 문서 코퍼스
- ❌ `data/index.pkl` - 사전 구축 인덱스
- ❌ `venv/` - 가상환경 폴더
- ❌ `__pycache__/` - Python 캐시
- ❌ `config/` - 설정 파일 (민감 정보 포함 가능)

### 2. 압축 파일 생성

#### Windows (PowerShell)
```powershell
# 제출용 폴더 생성
mkdir submission
copy PROJECT_REPORT.md submission/
copy README.md submission/
copy requirements.txt submission/
copy *.py submission/
xcopy /E /I src submission\src
xcopy /E /I results submission\results

# 압축 (7-Zip 또는 WinRAR 사용)
# submission 폴더를 압축
```

#### Linux/Mac
```bash
# 제출용 폴더 생성
mkdir submission
cp PROJECT_REPORT.md README.md requirements.txt *.py submission/
cp -r src submission/
cp -r results submission/

# 압축
cd submission
zip -r ../ir_project_submission.zip .
```

### 3. 압축 파일 내용 확인

압축 파일에는 다음이 포함되어야 합니다:
```
submission/
├── PROJECT_REPORT.md
├── README.md
├── requirements.txt
├── download_data.py
├── build_index.py
├── run_eval.py
├── app.py
├── src/
│   ├── __init__.py
│   ├── indexer.py
│   ├── tokenizer.py
│   ├── ranker.py
│   ├── tfidf_ranker.py
│   ├── searcher.py
│   ├── query_expander.py
│   ├── reranker.py
│   └── evaluator.py
└── results/
    ├── bm25_validation.txt
    ├── tfidf_validation.txt
    ├── hybrid_validation.txt
    ├── rerank_validation.txt
    └── summary_validation.txt
```

### 4. 실행 가능성 확인

제출 전 다음을 확인하세요:

1. **가상환경 없이도 실행 가능한지 확인**
   ```bash
   # 새 환경에서 테스트
   python download_data.py
   python build_index.py
   python run_eval.py --split validation
   streamlit run app.py
   ```

2. **의존성 확인**
   - `requirements.txt`에 모든 패키지 포함되어 있는지 확인

3. **경로 확인**
   - 상대 경로 사용 확인
   - 절대 경로 사용하지 않았는지 확인

### 5. 발표 준비

#### 발표 자료 (PDF)
- 시스템 구조 설명
- 주요 구현 내용
- 성능 실험 결과
- 데모 시나리오

#### 데모 준비
- 인덱스 사전 구축: `python build_index.py`
- 데모 쿼리 준비
- UI 테스트

### 6. 최종 확인

- [ ] 모든 소스 코드 원본 구현 확인
- [ ] 외부 검색 라이브러리 미사용 확인
- [ ] 프로젝트 보고서 완성 확인
- [ ] TREC 결과 파일 생성 확인
- [ ] 압축 파일 크기 확인 (너무 크지 않은지)
- [ ] 압축 파일 이름: `ir_project_[학번]_[이름].zip`

## 제출 일정

- **제출 마감**: 2025년 12월 22일 23:59
- **발표 및 데모**: 2025년 12월 23일 14:00

## 문의사항

프로젝트 관련 문의:
- 이메일: hyunje.song@jbnu.ac.kr
- 방문: 공대7호관 432호, 434호




