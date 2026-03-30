# 03_Common Code 최적화

## 3-1. 데이터 처리 코드 (data_*)

| 파일 | 주요 변경 |
|---|---|
| `Data_Datetime.py` | `list_invalidDate` → 벡터화 (`pd.to_datetime(errors='coerce')`), `create_col_weekdays` → `.dt.weekday`, `conv_midnight_24to00` → 벡터화, `calc_df_dt` 버그 수정 |
| `Data_Analysis.py` | `find_outlier_Usages` → IQR 벡터화, `print_residual` → 범용 `formula` 파라미터 추가, `calc_regression_metrics` 신규 |
| `Data_Preprocessing.py` | `del_outlier_Usages` → `.values` 배열 직접 조작으로 `SettingWithCopyWarning` 해소 |
| `Data_Visualization.py` | `visualization_df` → 범용 `plot_timeseries`로 리팩토링 (특정 컬럼 의존 제거) |
| `Data_Clustering.py` | 반복 플롯 로직 → `_plot_metric_by_k` 헬퍼 추출, K-range 정규화 중앙화 |

## 3-2. 데이터 제공처별 코드

| 파일 | 주요 변경 |
|---|---|
| `KASI_Holiday.py` | 3개 유사 API 함수 → `_fetch_special_days` 내부 헬퍼 + 3개 퍼블릭 래퍼로 통합 |
| `KDHC_Usage.py` | 잘못 배치된 `KMA_ASOS_DATA` 함수 → `KMA_Weather.py`로 이동, API URL/버전 상수 추출 |
| `KECO_AirKor.py` | 잘못 배치된 `Interpolate_KMA_ASOS` 함수 제거, 컬럼 리네이밍 맵/오염물질 컬럼 상수화 |
| `KIER_Usage.py` | `.iloc` 루프 → `pd.Series.diff()` 벡터화 |
| `KIER_Usage_M02.py` | 하드코딩 매핑 → 모듈 레벨 상수(`_DOMAIN`, `_COL_ACCU` 등) 추출, 들여쓰기 수정 |
| `KMA_Weather.py` | `pd.datetime`(deprecated) → `datetime.datetime.now()`, 컬럼 리네이밍/보간 전략 상수화, 보간 로직 벡터화 |
| `KorEx_Traffic.py` | 3개 API 함수 중복 로직 → `_call_api` 내부 헬퍼로 통합 |

## 3-3. 외부 라이브러리

| 파일 | 주요 변경 |
|---|---|
| `cluster_eval.py` | 거리 계산 이중 루프 → `scipy.spatial.distance.pdist`/`cdist` 벡터화, 미사용 함수 제거 |
| `Ref_deepant.py` | import 정리, 클래스 메서드 네이밍 정리 후 유지 |
| `Ref_deepant_utils.py` | `range` 파라미터 → `range_`로 빌트인 충돌 해소, import 정리 후 유지 |

## 3-4. 모델 공용 코드

| 파일 | 주요 변경 |
|---|---|
| `Common_Model_ML.py` | import 30줄 → 10줄 정리, KFold 집계 → `_aggregate_kfold_scores` 헬퍼 추출 |
| `Common_Model_DL.py` | import 110줄 → 8줄 정리, ML 중복 함수 4개 제거 → `Common_Model_ML`에서 import |

## 이점
- **성능**: 루프 → 벡터화로 대용량 데이터 처리 속도 향상
- **안정성**: 버그 수정 (`calc_df_dt`, `SettingWithCopyWarning`, deprecated API)
- **유지보수**: 함수 위치 정상화, 중복 제거, 상수 추출로 변경 포인트 단일화
- **가독성**: import 블록 대폭 축소 (모듈당 평균 30줄 → 8줄)
