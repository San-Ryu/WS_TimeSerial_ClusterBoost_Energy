"""
Common_Model_ML.py
ML 모델 (CB / DT / LGBM / RF / XGB) 학습·예측·평가 공통 모듈
+ 공용 지표(metrics) · 시각화 (DL 모듈에서도 import)

History
  2024-04-05  Created
  2026-03-30  Refactored - import 정리, KFold 중복 제거, 공용 함수 정리
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ── Font ────────────────────────────────────────────────────────────────────
_FONT_PATH = Path(__file__).parent / "Times New Roman.ttf"
FONT_TIMES_NEW_ROMAN = fm.FontProperties(fname=str(_FONT_PATH)) if _FONT_PATH.exists() else None

# ── 모델 매핑 ───────────────────────────────────────────────────────────────
_MODEL_NAME = {0: "CB", 1: "DT", 2: "LGBM", 3: "RF", 4: "XGB"}

_METRIC_KEYS = ["MAE", "MAPE", "MSE", "RMSE", "MSLE", "MBE", "R2"]


# =========================================================================== #
#  공용: Metrics
# =========================================================================== #

def mean_bias_error(y_true, y_pred) -> float:
    return float(np.sum(y_true - y_pred) / np.size(y_true))


def model_sk_metrics(y_true, y_pred) -> list[float]:
    """MAE, MAPE, MSE, RMSE, MSLE, MBE, R2 를 list로 반환."""
    mae = round(mean_absolute_error(y_true, y_pred), 4)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), 4)
    mse = round(mean_squared_error(y_true, y_pred), 4)
    rmse = round(np.sqrt(mse), 4)
    try:
        msle = round(mean_squared_log_error(y_true, y_pred), 4)
    except ValueError:
        msle = np.nan
    mbe = round(mean_bias_error(y_true, y_pred), 4)
    r2 = round(r2_score(y_true, y_pred), 4)
    return [mae, mape, mse, rmse, msle, mbe, r2]


# =========================================================================== #
#  공용: Visualization
# =========================================================================== #

def model_visualization(y_true, y_pred, title: str,
                        save_path: str | None = None) -> None:
    """Actual vs Predicted 비교 시각화."""
    font_kw = {"fontproperties": FONT_TIMES_NEW_ROMAN} if FONT_TIMES_NEW_ROMAN else {}
    plt.figure(figsize=(50, 15), dpi=500)
    plt.plot(y_true, color="red", label="Actual")
    plt.plot(y_pred, color="blue", label="Predicted")
    plt.title(title, fontsize=80, **font_kw)
    plt.xlabel("Period", fontsize=50, **font_kw)
    plt.xticks(fontsize=30)
    plt.ylabel("Energy Usage", fontsize=50, **font_kw)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=50)
    plt.savefig(save_path or title, dpi=500)
    plt.show()


# =========================================================================== #
#  공용: KFold 평균 계산 헬퍼
# =========================================================================== #

def _aggregate_kfold_scores(fold_scores: list[list[float]],
                            fold_times: list[float]):
    """
    각 fold의 [MAE, MAPE, MSE, RMSE, MSLE, MBE, R2]와 실행시간을
    평균·이력으로 정리해 반환.
    """
    arr = np.array(fold_scores)                   # (n_folds, 7)
    means = np.nanmean(arr, axis=0).round(4)      # (7,)
    avg_time = round(np.mean(fold_times), 4)

    kf_scores = list(means) + [avg_time]           # 8 values
    kf_hists = [arr[:, i].tolist() for i in range(arr.shape[1])] + [fold_times]
    return kf_scores, kf_hists


# =========================================================================== #
#  ML: Data Split
# =========================================================================== #

def data_train_test_split(df: pd.DataFrame, test_ratio: float,
                          target_col: str):
    """시계열 순서 유지 train/test 분할 → (train_X, train_Y, test_X, test_Y)."""
    train, test = train_test_split(df, test_size=test_ratio, shuffle=False)
    train_x, train_y = train.drop(columns=[target_col]), train[[target_col]]
    test_x, test_y = test.drop(columns=[target_col]), test[[target_col]]
    return train_x, train_y, test_x, test_y


# =========================================================================== #
#  ML: Predict
# =========================================================================== #

def model_ml_predict(train_x, train_y, test_x, test_y,
                     model_id: int):
    """
    ML 모델 학습·예측.

    model_id : 0=CB, 1=DT, 2=LGBM, 3=RF, 4=XGB
    Returns  : (y_actual, y_preds, elapsed_sec)
    """
    t0 = time.time()

    if model_id == 0:
        model = CatBoostRegressor(
            iterations=500, max_ctr_complexity=6, random_seed=10,
            od_type="Iter", od_wait=25, verbose=1000,
            depth=5, learning_rate=0.03,
        ).fit(train_x, train_y, cat_features=[], eval_set=[(train_x, train_y)])
    elif model_id == 1:
        model = DecisionTreeRegressor(max_depth=8).fit(train_x, train_y)
    elif model_id == 2:
        model = LGBMRegressor(
            n_estimators=10000, learning_rate=0.01, verbose=0,
        ).fit(train_x, train_y, eval_metric="mae", eval_set=[(train_x, train_y)])
    elif model_id == 3:
        model = RandomForestRegressor(
            max_depth=8, min_samples_leaf=8, min_samples_split=8, n_estimators=200,
        ).fit(train_x, train_y)
    elif model_id == 4:
        model = xgb.XGBRegressor(n_estimators=1000).fit(
            train_x, train_y,
            eval_set=[(test_x, test_y)],
            early_stopping_rounds=50, verbose=False,
        )
    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    elapsed = time.time() - t0
    preds = model.predict(test_x).reshape(-1, 1)
    actual = test_y.to_numpy().reshape(-1, 1)
    return actual, preds, elapsed


# =========================================================================== #
#  ML: Single Analysis
# =========================================================================== #

def model_ml_analysis_single(df: pd.DataFrame, model_id: int,
                             test_ratio: float, target_col: str):
    train_x, train_y, test_x, test_y = data_train_test_split(df, test_ratio, target_col)
    return model_ml_predict(train_x, train_y, test_x, test_y, model_id)


# =========================================================================== #
#  ML: KFold Analysis
# =========================================================================== #

def model_ml_analysis_with_KFold(df: pd.DataFrame, model_id: int,
                                 test_ratio: float, target_col: str,
                                 n_folds: int, shuffle: bool = False):
    """
    K-Fold 교차검증 ML 분석.

    Returns
    -------
    kf_scores : [MAE, MAPE, MSE, RMSE, MSLE, MBE, R2, AvgTime] 평균값
    kf_hists  : 각 지표별 fold 이력 리스트
    """
    kf = KFold(n_splits=n_folds, shuffle=shuffle)
    fold_scores, fold_times = [], []

    for train_idx, test_idx in kf.split(df):
        train_x = df.iloc[train_idx].drop(columns=[target_col])
        train_y = df.iloc[train_idx][[target_col]]
        test_x = df.iloc[test_idx].drop(columns=[target_col])
        test_y = df.iloc[test_idx][[target_col]]

        actual, preds, elapsed = model_ml_predict(train_x, train_y, test_x, test_y, model_id)
        fold_scores.append(model_sk_metrics(actual, preds))
        fold_times.append(elapsed)

    return _aggregate_kfold_scores(fold_scores, fold_times)
