"""
KIER_Usage.py
한국에너지기술연구원(KIER) 에너지 사용량 데이터 처리

History
  2023-11-14  Created
  2026-03-30  Refactored - .iloc 루프 → .diff() 벡터화, import 정리
"""

import pandas as pd


def create_inst_usage(df: pd.DataFrame, col_accu: str,
                      col_inst: str) -> pd.DataFrame:
    """
    적산(누적) 데이터로부터 순시(시점간 차이) 데이터 생성.
    shift(-1)을 사용해 다음 행과의 차이를 계산 (원본과 동일한 방향).

    Parameters
    ----------
    col_accu : 적산값 컬럼명
    col_inst : 생성할 순시값 컬럼명
    """
    df[col_inst] = df[col_accu].shift(-1) - df[col_accu]
    return df
