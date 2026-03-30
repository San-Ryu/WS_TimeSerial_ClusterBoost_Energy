# 04_Common Code 작명 체계 (PEP8 snake_case)

## 변경 사항

### 파일 리네이밍 (17개) — 접두사 기반 4-카테고리 체계

| 카테고리 | 변경 전 | 변경 후 |
|---|---|---|
| `data_` | `Data_Datetime.py` 외 4개 | `data_datetime.py` 외 4개 |
| `provider_` | `KASI_Holiday.py` 외 6개 | `provider_kasi.py` 외 6개 |
| `model_` | `Common_Model_ML.py`, `_DL.py` | `model_ml.py`, `model_dl.py` |
| `ref_` | `cluster_eval.py`, `Ref_deepant*.py` | `ref_cluster_eval.py`, `ref_deepant*.py` |

### 함수/상수명 통일
- `get_Dunn_index()` → `get_dunn_index()`
- `Font_TimesNewRoman` → `FONT_TIMES_NEW_ROMAN`

### import 경로 수정
- `.py` 내부 cross-import 3개
- `.ipynb` 노트북 90개 (DEV_ 접두사 레거시 포함 일괄 치환)

### 디렉터리 리네이밍
- `Src_Dev_Common/` → `core/`

## 이점
- **PEP8 완전 준수**: 모듈명 소문자 + 밑줄 표준
- **탐색기 자동 정렬**: 같은 카테고리 파일이 그룹으로 모여 표시
- **자동완성 효율**: `data_` / `provider_` / `model_` / `ref_` 입력으로 카테고리별 즉시 필터
- **의미 명확**: 파일명만으로 역할(데이터처리/제공처/모델/참조)을 즉시 파악
