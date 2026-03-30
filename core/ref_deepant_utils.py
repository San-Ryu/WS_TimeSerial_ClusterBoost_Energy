"""
Ref_deepant_utils.py
DeepAnT 시각화 유틸리티

Reference
  Git : https://github.com/datacubeR/DeepAnt/
History
  2023-11-13  Created (외부 코드 복제)
  2026-03-30  Refactored - 코드 정리
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_predictions(preds, threshold, bins=80):
    """Loss 분포 히스토그램 + 임계값 표시."""
    sns.displot(preds, bins=bins, kde=True, height=8, aspect=2)
    plt.axvline(x=threshold, color="r", linestyle="--", label="Chosen Threshold")
    plt.title("Loss Distribution")
    plt.legend()


def loss_plot(preds, threshold):
    """Loss 시계열 + 임계값 수평선."""
    preds.plot(figsize=(15, 8), title="Chosen Threshold", label="Loss")
    plt.axhline(y=threshold, color="r", linestyle="--", label="Chosen Threshold")
    plt.legend()
    plt.show()


def ts_plot(df, preds, threshold, alg="DeepAnT", range_=None):
    """시계열에 이상치 탐지 결과를 scatter로 표시."""
    idx = preds.loc[lambda x: x > threshold].index
    plt.figure(figsize=(20, 8))
    if range_ is not None:
        lo, hi = range_
        df = df[lo:hi]
    plt.plot(df, label="_nolegend_")
    plt.scatter(idx, df.loc[idx], color="red",
                label=f"Detected Anomalies by {alg}")
    plt.title("Detected Anomalies in Time Series")
    plt.legend()
    plt.show()
