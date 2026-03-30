# 06_소스 디렉터리 정리

## 6-1. KIRBS Ver 중복 제거

- `Src_Energy_M02`와 `Src_Energy_M02_KIRBS Ver`을 MD5 해시로 비교
- **84개 파일**: 내용 완전 동일 → `KIRBS Ver` 삭제
- **2개 파일**: 사실상 동일
  - `M02-01_Data_04_*.ipynb`: KIRBS에만 `!pip install` 환경설정 셀 추가 (분석 코드 동일)
  - `M02-03_Model_Seq2Seq-01_*.ipynb`: 커널 메타데이터 12B 차이 (코드/출력 동일)
- `Src_Energy_M02_KIRBS Ver/` 전체 삭제

## 6-2. 디렉터리 리네이밍 및 임시 파일 정리

- `Src_Energy_M02/` → `src/` 리네이밍
- `catboost_info/` 삭제 (CatBoost 학습 시 자동 재생성되는 로그/임시 파일)
- `temp.csv`, `tmp` 삭제 (임시 파일)

## 이점
- 중복 디렉터리 제거로 디스크 용량 절감 및 혼동 방지
- 간결한 디렉터리명으로 프로젝트 탐색 효율 향상

---

## 최종 프로젝트 구조

```
WS_TimeSerial_ClusterBoost_Energy/
├── core/            # 공용 모듈 (17개 .py)
│   ├── data_*.py         # 데이터 처리 (5개)
│   ├── provider_*.py     # 데이터 제공처 (7개)
│   ├── model_*.py        # 모델 공용 (2개)
│   └── ref_*.py          # 외부 참조 (3개)
├── data/            # 데이터셋
│   ├── KMA_ASOS/         # 기상청 ASOS 데이터
│   └── KIER_Usage/       # KIER 사용량 데이터
├── results/         # 실험 결과물
├── src/             # 메인 소스 (노트북)
├── cursor/          # 프로젝트 문서
├── README.md
└── requirements.txt
```
