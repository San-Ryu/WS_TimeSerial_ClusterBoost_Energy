"""
Data_Analysis.py
데이터 분석 (기술통계, 이상치 탐지, 회귀 잔차) 공통 모듈

History
  2023-11-14  Created
  2026-03-30  Refactored - 벡터화, 하드코딩 제거, import 정리
"""

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


# --------------------------------------------------------------------------- #
#  Descriptive Statistics
# --------------------------------------------------------------------------- #

def print_desc_statistic(df: pd.DataFrame, col: str) -> None:
    """기술통계량(Min/Std/Median/Mean/Max) 출력 후 IQR 이상치 탐지."""
    s = df[col]
    print("=" * 50)
    print(f"  Min    : {s.min()}")
    print(f"  Std    : {s.std()}")
    print(f"  Median : {s.median()}")
    print(f"  Mean   : {s.mean()}")
    print(f"  Max    : {s.max()}")
    print("=" * 50)

    outlier_rows = find_outlier_iqr(df, col)
    print(f"  Outlier rows: {len(outlier_rows)}")
    print("=" * 50)


# --------------------------------------------------------------------------- #
#  Outlier Detection (IQR)
# --------------------------------------------------------------------------- #

def find_outlier_iqr(df: pd.DataFrame, col: str,
                     q_low: float = 0.25, q_high: float = 0.90,
                     coef: float = 1.5) -> list[int]:
    """
    IQR 방식으로 이상치 행 위치(정수 인덱스)를 반환.

    Parameters
    ----------
    q_low / q_high : 분위수 경계 (기본 0.25 / 0.90)
    coef           : IQR 배수 (기본 1.5)
    """
    s = df[col]
    q1, q3 = s.quantile(q_low), s.quantile(q_high)
    iqr = q3 - q1
    lower, upper = q1 - coef * iqr, q3 + coef * iqr

    print(f"  IQR Range — Upper: {upper:.4f}, Q3: {q3:.4f}, "
          f"IQR: {iqr:.4f}, Median: {s.median():.4f}, "
          f"Q1: {q1:.4f}, Lower: {lower:.4f}")

    mask = (s > upper) | (s < lower) | (s < 0)
    outlier_idx = list(np.where(mask)[0])
    print(f"  cnt_outlier = {len(outlier_idx)}")
    return outlier_idx


# --------------------------------------------------------------------------- #
#  Residual Analysis (OLS)
# --------------------------------------------------------------------------- #

def print_residual(df: pd.DataFrame, formula: str) -> None:
    """
    OLS 잔차분석 결과 출력.

    Parameters
    ----------
    formula : statsmodels 패턴 (예: 'y ~ x1 + x2')
    """
    result = ols(formula, data=df).fit()
    print(result.summary())


# --------------------------------------------------------------------------- #
#  Regression Metrics (convenience wrapper)
# --------------------------------------------------------------------------- #

def calc_regression_metrics(y_true, y_pred) -> dict:
    """MAE, MAPE, MSE, RMSE, R² 를 dict로 반환."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }
