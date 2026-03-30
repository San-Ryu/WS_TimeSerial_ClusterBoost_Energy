# WS_TimeSerial_ClusterBoost_Energy

시계열 기반 에너지 소비 예측 프로젝트.
K-Means 클러스터링으로 건물 사용 패턴을 분류하고, ML/DL 모델로 전력 사용량을 예측한다.

Time-series energy consumption forecasting.
Classifies building usage patterns via K-Means clustering, then predicts electricity demand using ML and DL models with KMA weather and KIER usage data.

## Project Structure

```
├── core/            # 공용 모듈 (17개 .py)
│   ├── data_*           # 데이터 처리
│   ├── provider_*       # 데이터 제공처 (KMA, KIER, KASI, KDHC, KECO, KorEx)
│   ├── model_*          # 모델 공용 (ML / DL)
│   └── ref_*            # 외부 참조 (DeepAnT, Dunn Index)
├── data/            # 데이터셋
│   ├── KMA_ASOS/        # 기상청 ASOS 기상 데이터
│   └── KIER_Usage/      # KIER 전력 사용 데이터
├── results/         # 실험 결과물
├── src/             # 메인 소스 (Jupyter 노트북)
├── cursor/          # 리팩토링 변경 이력
├── requirements.txt
└── README.md
```

## Models

| 유형 | 모델 |
|---|---|
| ML | CatBoost, LightGBM, XGBoost, Random Forest, Decision Tree |
| DL | 1D-CNN LSTM, 1D-CNN Seq2Seq |

## Data Sources

| 제공처 | 데이터 |
|---|---|
| KMA (기상청) | ASOS 시간별 기상 관측 |
| KIER (에너지연구원) | 건물별 전력 사용량 |
| KASI (천문연구원) | 공휴일 정보 |
| KECO (환경공단) | AirKorea 대기질 |
| KDHC (지역난방공사) | 열 공급량 |

## Setup

```bash
pip install -r requirements.txt
```
