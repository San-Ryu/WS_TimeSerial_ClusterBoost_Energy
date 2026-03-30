"""
KECO_AirKor.py
한국환경공단 에어코리아 대기질 데이터 정제·보간

History
  2023-05-25  Created
  2023-05-26  기상청/한국환경공단 수집 기능 분리
  2023-11-17  공통코드 사용 및 간략화
  2026-03-30  Refactored - KMA 관련 함수 KMA_Weather.py로 이관, 보간 벡터화
"""

import pandas as pd

_RENAME_MAP = {
    "지역":     "REGION",
    "측정소명": "NM_OBSERVATORY",
    "측정소코드": "CD_OBSERVATORY",
    "측정일시": "METER_DATE",
}

_COLS_FINAL = [
    "METER_DATE",
    "REGION", "CD_OBSERVATORY", "NM_OBSERVATORY",
    "SO2", "CO", "O3", "NO2", "PM10",
]

_POLLUTANT_COLS = ["SO2", "CO", "O3", "NO2", "PM10"]


def rename_airkor(df: pd.DataFrame) -> pd.DataFrame:
    """에어코리아 원시 컬럼명 → 영문 표준명 변환."""
    return df.rename(columns=_RENAME_MAP)[_COLS_FINAL]


def interpolate_airkor(df: pd.DataFrame) -> pd.DataFrame:
    """대기질 수치 컬럼 선형 보간."""
    df["METER_DATE"] = pd.to_datetime(df["METER_DATE"])
    df[_POLLUTANT_COLS] = df[_POLLUTANT_COLS].interpolate()
    return df
