"""
KASI_Holiday.py
한국천문연구원 특일정보 API (국경일 / 공휴일 / 기념일)

History
  2023-11-17  Created
  2026-03-30  Refactored - 중복 함수 통합, import 정리
"""

import json
import urllib
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus

import pandas as pd

_BASE_URL = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService"

_ENDPOINT = {
    "holiday":     f"{_BASE_URL}/getHoliDeInfo",
    "rest":        f"{_BASE_URL}/getRestDeInfo",
    "anniversary": f"{_BASE_URL}/getAnniversaryInfo",
}


def _fetch_special_days(year: int, api_key: str,
                        kind: str) -> pd.DataFrame:
    """
    공통 API 호출.

    Parameters
    ----------
    kind : 'holiday' | 'rest' | 'anniversary'
    """
    url = _ENDPOINT[kind]
    params = "?" + urlencode({
        quote_plus("ServiceKey"): api_key,
        quote_plus("_type"):      "json",
        quote_plus("solYear"):    str(year),
        quote_plus("numOfRows"):  100,
    })

    req = urllib.request.Request(url + unquote(params))
    body = urlopen(req, timeout=600).read()
    items = json.loads(body)["response"]["body"]["items"]["item"]
    return pd.DataFrame(items)


def fetch_holidays(year: int, api_key: str) -> pd.DataFrame:
    """국경일 정보 조회."""
    return _fetch_special_days(year, api_key, "holiday")


def fetch_rest_days(year: int, api_key: str) -> pd.DataFrame:
    """공휴일 정보 조회."""
    return _fetch_special_days(year, api_key, "rest")


def fetch_anniversaries(year: int, api_key: str) -> pd.DataFrame:
    """기념일 정보 조회."""
    return _fetch_special_days(year, api_key, "anniversary")
