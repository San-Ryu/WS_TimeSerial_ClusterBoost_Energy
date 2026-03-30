"""
Data_Visualization.py
데이터 시각화 공통 모듈

History
  2023-11-14  Created
  2026-03-30  Refactored - 벡터화, import 정리, 함수 범용화
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 – 모듈 로드 시 스타일 적용

plt.rcParams["figure.figsize"] = [10, 8]


# --------------------------------------------------------------------------- #
#  Time-Series Line Plot
# --------------------------------------------------------------------------- #

def plot_timeseries(df: pd.DataFrame, col_date: str, col_value: str,
                    color: str = "tab:blue", title: str | None = None,
                    figsize: tuple = (30, 5)) -> None:
    """
    시계열 라인 차트.

    Parameters
    ----------
    col_date  : datetime 컬럼명
    col_value : 시각화할 값 컬럼명
    color     : 라인 색상
    title     : 그래프 제목 (기본: col_value)
    figsize   : 그래프 크기
    """
    dates = pd.to_datetime(df[col_date])
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title or col_value, fontsize=20, fontweight="bold", pad=20)
    ax.plot(dates, df[col_value], color=color)
    plt.tight_layout()
    plt.show()
