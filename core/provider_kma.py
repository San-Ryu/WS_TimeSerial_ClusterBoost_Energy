"""
KMA_Weather.py
대한민국 기상청 종관기상관측(ASOS) 데이터 수집·정제·보간

History
  2023-05-25  Created
  2023-11-17  공통코드 사용 및 간략화
  2024-03-14  Rename_KMA_ASOS_CSVDOWN / Interpolate 추가
  2026-03-30  Refactored - pd.datetime 제거, 보간 벡터화, 오배치 함수 통합
"""

import json
import urllib
from datetime import datetime
from urllib.parse import urlencode, unquote, quote_plus
from urllib.request import urlopen

import pandas as pd

# ── ASOS 컬럼 매핑 (API / CSV 공용) ────────────────────────────────────────
_ASOS_COLS_FINAL = [
    "METER_DATE",
    "temp_outdoor", "temp_dew_point", "temp_ground",
    "humidity",
    "rainfall", "snowfall", "snowfall_3hr",
    "wind_speed", "wind_direction",
    "pressure_vapor", "pressure_area", "pressure_sea",
    "sunshine", "solar_radiation",
    "cloud_total", "cloud_midlow",
    "visual_range",
]

_RENAME_API = {
    "tm": "METER_DATE",
    "ta": "temp_outdoor", "td": "temp_dew_point", "ts": "temp_ground",
    "hm": "humidity", "rn": "rainfall", "dsnw": "snowfall", "hr3Fhsc": "snowfall_3hr",
    "ws": "wind_speed", "wd": "wind_direction",
    "pv": "pressure_vapor", "pa": "pressure_area", "ps": "pressure_sea",
    "ss": "sunshine", "icsr": "solar_radiation",
    "dc10Tca": "cloud_total", "dc10LmcsCa": "cloud_midlow",
    "vs": "visual_range",
}

_RENAME_CSV = {
    "일시": "METER_DATE",
    "기온(°C)": "temp_outdoor", "이슬점온도(°C)": "temp_dew_point", "지면온도(°C)": "temp_ground",
    "습도(%)": "humidity", "강수량(mm)": "rainfall", "적설(cm)": "snowfall", "3시간신적설(cm)": "snowfall_3hr",
    "풍속(m/s)": "wind_speed", "풍향(16방위)": "wind_direction",
    "증기압(hPa)": "pressure_vapor", "현지기압(hPa)": "pressure_area", "해면기압(hPa)": "pressure_sea",
    "일조(hr)": "sunshine", "일사(MJ/m2)": "solar_radiation",
    "전운량(10분위)": "cloud_total", "중하층운량(10분위)": "cloud_midlow",
    "시정(10m)": "visual_range",
}

# 보간 전략: interpolate 대상 / fillna(0) 대상
_INTERP_COLS = [
    "temp_outdoor", "temp_dew_point", "temp_ground",
    "humidity",
    "wind_speed", "wind_direction",
    "pressure_vapor", "pressure_area", "pressure_sea",
]
_FILLZERO_COLS = [
    "rainfall", "snowfall", "snowfall_3hr",
    "sunshine", "solar_radiation",
    "cloud_total", "cloud_midlow",
    "visual_range",
]


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #

def fetch_asos(observatory: str, api_key: str, year: int,
               interval: str = "HR", page: int = 1) -> pd.DataFrame:
    """
    종관기상관측(ASOS) OpenAPI 호출.

    Parameters
    ----------
    observatory : 관측소 코드 (예: '108')
    interval    : 'HR'(시간) / 'DAY'(일)
    page        : 페이지 번호
    """
    now = datetime.now()
    if year == now.year:
        end_date = f"{year}{now.month:02d}{now.day:02d}"
    else:
        end_date = f"{year}1231"

    url = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
    params = "?" + urlencode({
        quote_plus("ServiceKey"): api_key,
        quote_plus("pageNo"):     page,
        quote_plus("numOfRows"):  "999",
        quote_plus("dataType"):   "JSON",
        quote_plus("dataCd"):     "ASOS",
        quote_plus("dateCd"):     interval,
        quote_plus("startDt"):    f"{year}0101",
        quote_plus("startHh"):    "00",
        quote_plus("endDt"):      end_date,
        quote_plus("endHh"):      "23",
        quote_plus("stnIds"):     observatory,
    })

    req = urllib.request.Request(url + unquote(params))
    body = urlopen(req, timeout=60).read()
    items = json.loads(body)["response"]["body"]["items"]["item"]
    return pd.DataFrame(items)


# --------------------------------------------------------------------------- #
#  Rename
# --------------------------------------------------------------------------- #

def rename_asos_api(df: pd.DataFrame) -> pd.DataFrame:
    """API 응답 컬럼명 → 영문 표준명으로 변환."""
    return df.rename(columns=_RENAME_API)[_ASOS_COLS_FINAL]


def rename_asos_csv(df: pd.DataFrame) -> pd.DataFrame:
    """CSV 다운로드 컬럼명(한글) → 영문 표준명으로 변환."""
    return df.rename(columns=_RENAME_CSV)[_ASOS_COLS_FINAL]


# --------------------------------------------------------------------------- #
#  Interpolation
# --------------------------------------------------------------------------- #

def interpolate_asos(df: pd.DataFrame) -> pd.DataFrame:
    """
    ASOS 기상 데이터 보간.
    - 연속성 컬럼(기온/습도/기압 등) → 선형 보간
    - 이벤트성 컬럼(강수/적설/일사 등) → 0 채움
    """
    df["METER_DATE"] = pd.to_datetime(df["METER_DATE"])
    df[_INTERP_COLS] = df[_INTERP_COLS].interpolate()
    df[_FILLZERO_COLS] = df[_FILLZERO_COLS].fillna(0)
    return df
