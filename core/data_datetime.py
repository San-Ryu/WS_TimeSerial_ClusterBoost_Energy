"""
Data_Datetime.py
날짜/시간 데이터 처리 공통 모듈

History
  2023-11-14  Created
  2026-03-30  Refactored - 벡터화, 버그 수정, import 정리
"""

import datetime as dt
from datetime import timedelta

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Validation
# --------------------------------------------------------------------------- #

def validate_date(date_text: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> bool:
    """단일 날짜 문자열이 지정 포맷에 유효한지 검사."""
    try:
        dt.datetime.strptime(date_text, fmt)
        return True
    except (ValueError, TypeError):
        return False


def list_invalid_dates(df: pd.DataFrame, col: str,
                       fmt: str = "%Y-%m-%d %H:%M:%S") -> list[int]:
    """
    DataFrame의 날짜 컬럼에서 유효하지 않은 행 인덱스를 반환.
    pd.to_datetime(errors='coerce')를 사용해 벡터 처리.
    """
    parsed = pd.to_datetime(df[col], format=fmt, errors="coerce")
    invalid_mask = parsed.isna() & df[col].notna()
    invalid_idx = list(df.index[invalid_mask])
    print(f"Total rows: {len(df)}, Invalid dates: {len(invalid_idx)}")
    return invalid_idx


# --------------------------------------------------------------------------- #
#  Column Creation
# --------------------------------------------------------------------------- #

def create_col_ymdhm(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """날짜 컬럼을 YEAR, MONTH, DAY, HOUR, MINUTE 개별 컬럼으로 분해."""
    df[col] = pd.to_datetime(df[col])
    df["YEAR"] = df[col].dt.year
    df["MONTH"] = df[col].dt.month
    df["DAY"] = df[col].dt.day
    df["HOUR"] = df[col].dt.hour
    df["MINUTE"] = df[col].dt.minute
    return df


def create_col_weekdays(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """요일명 및 요일코드(0=Mon … 6=Sun) 컬럼 추가."""
    df["day_of_the_week"] = df[col].dt.day_name()
    df["code_day_of_the_week"] = df[col].dt.weekday
    return df


def create_col_datetime(df: pd.DataFrame, col_tar: str,
                        col_y: str, col_m: str, col_d: str,
                        col_h: str | None = None,
                        col_min: str | None = None,
                        col_s: str | None = None) -> pd.DataFrame:
    """YEAR/MONTH/DAY/HOUR/MINUTE/SECOND 컬럼을 합쳐 datetime 컬럼 생성."""
    h = df[col_h].astype(int) if col_h else 0
    m = df[col_min].astype(int) if col_min else 0
    s = df[col_s].astype(int) if col_s else 0

    df.insert(0, col_tar, pd.to_datetime(
        pd.DataFrame({
            "year": df[col_y].astype(int),
            "month": df[col_m].astype(int),
            "day": df[col_d].astype(int),
            "hour": h, "minute": m, "second": s,
        })
    ))
    return df


def create_col_week(df: pd.DataFrame, col: str,
                    type_week: str = "Y-W") -> pd.DataFrame:
    """ISO Week 컬럼 추가. type_week: 'Y-W' → '2024-03', 'W' → '03'."""
    fmt = "%G-%V" if type_week == "Y-W" else "%V"
    df["WEEK"] = df[col].dt.strftime(fmt)
    return df


# --------------------------------------------------------------------------- #
#  DataFrame Generation
# --------------------------------------------------------------------------- #

def create_df_dt(df: pd.DataFrame, col: str,
                 dt_start, dt_end, interval: str) -> pd.DataFrame:
    """
    지정 기간·간격의 시계열 DataFrame 생성.
    YEAR/MONTH/DAY/HOUR/MINUTE + 요일 컬럼 자동 부여.
    """
    df[col] = pd.date_range(start=str(dt_start), end=str(dt_end), freq=interval)
    df["day_of_the_week"] = df[col].dt.day_name()
    df = create_col_ymdhm(df, col)
    df = create_col_weekdays(df, col)
    return df


# --------------------------------------------------------------------------- #
#  Period Calculation
# --------------------------------------------------------------------------- #

def calc_df_dt(df1: pd.DataFrame, col: str,
               df2: pd.DataFrame | None = None):
    """
    DataFrame(들)에서 날짜 컬럼의 최솟값(시점)·최댓값(종점) 반환.
    df2가 주어지면 두 DataFrame의 합산 범위를 반환.
    """
    date1 = pd.to_datetime(df1[col])
    if df2 is None or df2.empty:
        return date1.min(), date1.max()

    date2 = pd.to_datetime(df2[col])
    return min(date1.min(), date2.min()), max(date1.max(), date2.max())


# --------------------------------------------------------------------------- #
#  24시 → 00시 변환
# --------------------------------------------------------------------------- #

def conv_midnight_24to00(df: pd.DataFrame, col_tar: str,
                         col_src: str, fmt: str) -> pd.DataFrame:
    """
    24:00 표기를 다음날 00:00으로 변환.
    벡터 처리: 먼저 일괄 파싱 → 실패 건만 보정.
    """
    raw = df[col_src].astype(str)
    parsed = pd.to_datetime(raw, format=fmt, errors="coerce")

    fail_mask = parsed.isna()
    if fail_mask.any():
        fixed = raw[fail_mask].str[:-2] + "00"
        parsed[fail_mask] = pd.to_datetime(fixed, format=fmt) + timedelta(days=1)

    df[col_tar] = parsed
    return df
