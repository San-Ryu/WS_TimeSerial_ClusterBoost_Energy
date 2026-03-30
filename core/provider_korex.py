"""
KorEx_Traffic.py
한국도로공사 교통데이터 API (톨게이트 목록 / 입출구 / 교통량)

History
  2023-11-17  Created
  2026-03-30  Refactored - 공통 호출 함수 추출, import 정리
"""

import json
import urllib
from urllib.parse import urlencode, unquote, quote_plus
from urllib.request import urlopen

import pandas as pd

_BASE_URL = "https://www.bigdata-transportation.kr/api"


def _call_api(api_key: str, params_dict: dict,
              result_key: str) -> pd.DataFrame:
    """공통 API 호출 헬퍼."""
    base_params = {quote_plus("apiKey"): api_key}
    base_params.update({quote_plus(k): v for k, v in params_dict.items()})
    query = "?" + urlencode(base_params)

    req = urllib.request.Request(_BASE_URL + unquote(query))
    body = urlopen(req, timeout=600).read()
    items = json.loads(body)["result"][result_key]
    return pd.DataFrame(items)


def fetch_tollgates(api_key: str) -> pd.DataFrame:
    """톨게이트 목록 현황 조회."""
    return _call_api(api_key, {
        "productId": "PRDTNUM_000000020307",
        "numOfRows": 999,
    }, result_key="unitLists")


def fetch_tollgate_in_out(api_key: str, tollgate_code: str,
                          in_out: str = "") -> pd.DataFrame:
    """영업소별 입구/출구 현황 조회."""
    return _call_api(api_key, {
        "productId": "PRDTNUM_000000020305",
        "unitCode":  tollgate_code,
        "inoutType": in_out,
    }, result_key="laneStatusVO")


def fetch_tollgate_traffic(api_key: str, unit_code: str,
                           tm_type: str = "1") -> pd.DataFrame:
    """
    톨게이트 입/출구 교통량 조회.

    Parameters
    ----------
    tm_type   : '1'(1시간) / '2'(15분)
    unit_code : 영업소 코드 (예: '111'=청주, '112'=남청주)
    """
    return _call_api(api_key, {
        "productId": "PRDTNUM_000000020308",
        "tmType":    str(tm_type),
        "unitCode":  str(unit_code),
        "numOfRows": 999,
    }, result_key="trafficIc")
