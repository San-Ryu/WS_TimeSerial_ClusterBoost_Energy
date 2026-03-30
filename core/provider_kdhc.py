"""
KDHC_Usage.py
한국지역난방공사 시간대별 열공급량 OpenAPI

History
  2023-11-29  Created
  2026-03-30  Refactored - 오배치 KMA 함수 제거, import 정리
"""

import json
import urllib
from urllib.parse import urlencode, unquote, quote_plus
from urllib.request import urlopen

import pandas as pd

_BASE_URL = "https://api.odcloud.kr/api/15099319/v1/"

_VERSION_MAP = {
    "v20181231": "uddi:4ccf1119-648f-4b4a-b6f8-f66499741f25",
    "v20211231": "uddi:87d90a27-4f90-4cf9-b0e8-bff7f352bfed",
    "v20220930": "uddi:ff86e691-7bf4-46b4-a828-e9ebda6aea1a",
}


def fetch_heat_usage(api_key: str, page: int = 1,
                     version: str = "v20220930") -> pd.DataFrame:
    """
    한국지역난방공사 시간대별 열공급량 조회.

    Parameters
    ----------
    page    : 페이지 번호
    version : API 데이터 버전 ('v20181231' / 'v20211231' / 'v20220930')
    """
    url = _BASE_URL + _VERSION_MAP[version]
    params = "?" + urlencode({
        quote_plus("serviceKey"):   api_key,
        quote_plus("page"):         page,
        quote_plus("perPage"):      "999",
        quote_plus("totalCount"):   0,
        quote_plus("currentCount"): 0,
        quote_plus("matchCount"):   0,
    })

    req = urllib.request.Request(url + unquote(params))
    body = urlopen(req, timeout=60).read()
    items = json.loads(body)["data"]
    return pd.DataFrame(items)
