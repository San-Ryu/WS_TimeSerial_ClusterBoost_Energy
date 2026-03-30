"""
KIER_Usage_M02.py
KIER M02 프로젝트 전용 – 도메인/디렉토리/파일명 매핑

History
  2024-04-03  Created
  2026-03-30  Refactored - import 정리, 들여쓰기 버그 수정
"""

# ── 도메인 매핑 테이블 ─────────────────────────────────────────────────────
_DOMAIN = {0: "ELEC", 1: "HEAT", 2: "WATER", 3: "HOT_HEAT", 4: "HOT_FLOW", 5: "GAS"}

_COL_ACCU = {
    0: "ACTUAL_ACCU_EFF", 1: "ACCU_HEAT", 2: "ACCU_FLOW",
    3: "ACCU", 4: "ACCU", 5: "ACCU_FLOW",
}
_COL_INST = {
    0: "INST_EFF", 1: "INST_HEAT", 2: "INST_FLOW",
    3: "INST", 4: "INST", 5: "INST_FLOW",
}

_INTERVAL = {0: "10MIN", 1: "1H", 2: "1D", 3: "1W", 4: "1M"}


def create_domain_str(domain_id: int) -> tuple[str, str, str]:
    """
    도메인 ID → (도메인명, 적산컬럼명, 순시컬럼명) 반환.
    """
    name = _DOMAIN[domain_id]
    col_accu = f"{name}_{_COL_ACCU[domain_id]}"
    col_inst = f"{name}_{_COL_INST[domain_id]}"
    print(f"{domain_id} : {name}")
    return name, col_accu, col_inst


def create_dir_str(domain: str) -> tuple[str, str, str, str, str]:
    """
    도메인명 → 관련 디렉토리 경로 반환.
    """
    base = "../data_Energy_KIER/"
    return (
        base,
        f"{base}KIER_0_Raw/",
        f"{base}KIER_1_Cleansed/",
        f"{base}KIER_2_BLD/",
        f"{base}KIER_3_H_{domain}/",
    )


def create_file_str(domain: str,
                    interval_id: int | None = None) -> tuple[str, str, str, str]:
    """
    도메인/인터벌 → (인터벌명, 원본파일명, 세대목록파일명, 리샘플파일명) 반환.
    """
    interval = _INTERVAL.get(interval_id, "10MIN") if interval_id is not None else "10MIN"

    file_raw = f"KIER_RAW_{domain}_2024-06-07.csv"
    file_hlist = "KIER_hList_common.csv"
    file_resampled = f"KIER_{domain}_INST_{interval}_Resampled.csv"

    print(f"str_fileRaw : {file_raw}")
    print(f"str_fileRaw_hList : {file_hlist}")
    print(f"str_file : {file_resampled}")

    return interval, file_raw, file_hlist, file_resampled
