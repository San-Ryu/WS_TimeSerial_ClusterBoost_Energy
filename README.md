# WS_TimeSerial_ClusterBoost_Energy

클러스터링 기반 ML 앙상블을 통한 건물 전력 소비 예측 연구 코드 저장소.

> Sim, T., Ryu, S., Lee, D., Lee, S., Chun, C.-J., & Moon, H. (2025).
> *A machine learning ensemble framework based on a clustering algorithm for improving electric power consumption performance.*
> **Scientific Reports**, 15(1), 40172. https://doi.org/10.1038/s41598-025-23978-w

---

## 연구 배경

정확한 전력 소비 예측은 사용자 편의성과 공급자 효율성 모두에 중요하다.
기존 ML 기반 예측 연구는 건물별 소비 패턴의 이질성을 충분히 반영하지 않아 정확도에 한계가 있었다.
본 연구는 **K-Means 클러스터링으로 건물을 소비 패턴 집단으로 분류**하고,
**각 집단에 최적화된 ML 모델을 조합한 앙상블 프레임워크**를 제안하여 예측 정확도를 개선한다.

---

## 연구 프레임워크

```
[데이터 수집]
  KMA ASOS 시간 기상 관측 + KIER 세대별 전력 사용량(10분 단위)
        │
        ▼
[전처리]
  Wide-format 변환 → 결측치 3단계 보간
  → IQR 이상치 제거 → 5개 시간 단위 리샘플링
  (10min / 1h / 1day / 1week / 1month)
        │
        ▼
[최적 K 탐색] ── 평가지표 4종 ──────────────────────────────
  Elbow Method / Silhouette Score /                         │
  Calinski-Harabasz Index / Dunn Index                      │
        │                                              결과: K=2 (1M)
        ▼
[K-Means 클러스터링]
  C0: 142세대 (저소비군)   C1: 206세대 (고소비군)
        │
        ▼
[클러스터별 ML 모델 학습 · KFold CV]
  CatBoost / LightGBM / XGBoost / Random Forest / Decision Tree
        │
        ▼
[앙상블 구성]
  각 클러스터의 최고 성능 모델 조합 → 4종
  CB-CB / CB-LGBM / LGBM-CB / LGBM-LGBM
        │
        ▼
[성능 평가]
  MAE / MSE / RMSE / R² + 통계 검정 (vs. Control Group)
```

---

## 주요 결과

| 구분 | 내용 |
|---|---|
| 최적 클러스터 수 | K = 2 (월별 리샘플링 기준) |
| 클러스터 구성 | C0 142세대 / C1 206세대 |
| 클러스터별 최고 성능 모델 | CatBoost, LightGBM |
| 앙상블 모델 | CB-CB · CB-LGBM · LGBM-CB · LGBM-LGBM |
| 통계 검정 결과 | 4종 앙상블 모두 Control Group 대비 유의미한 성능 향상 (p < 0.05 or 0.01) |

---

## 데이터 출처

| 제공처 | 데이터 | 용도 |
|---|---|---|
| KMA (기상청) | ASOS 시간별 종관기상관측 | 외부 기상 특성 |
| KIER (에너지기술연구원) | 세대별 전력·열·가스 사용량 (10분) | 예측 대상 |
| KASI (천문연구원) | 공휴일·특일 정보 | 달력 특성 |
| KECO (환경공단) | AirKorea 대기질 | 외부 환경 특성 |
| KDHC (지역난방공사) | 시간대별 열공급량 | 외부 공급 특성 |
| KorEx (한국도로공사) | 톨게이트 교통량 | 외부 인구 이동 특성 |

---

## 인용

```bibtex
@article{Sim2025,
  author   = {Sim, Taeyong and Ryu, Sanghyun and Lee, Dongjun
              and Lee, Sujin and Chun, Chang-Jae and Moon, Hyeonjoon},
  title    = {A machine learning ensemble framework based on a clustering
              algorithm for improving electric power consumption performance},
  journal  = {Scientific Reports},
  year     = {2025},
  volume   = {15},
  number   = {1},
  pages    = {40172},
  doi      = {10.1038/s41598-025-23978-w}
}
```

---

코드 구조 및 각 소스파일 설명 → [`desc_code.md`](desc_code.md)
