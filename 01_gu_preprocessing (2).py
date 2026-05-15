"""
서울시 야간 보행 안전 사각지대 분석 - 구 단위 데이터 전처리

변수명 규칙 (협업자 통일 규칙):
- crime_data      : 5대 범죄 발생 건수 (구 단위)
- crime_rate      : 범죄발생률 = crime_data / pop_tot × 10만 (파생변수)
- single_cnt      : 1인가구 수 (구 단위)
- pop_tot         : 야간 생활인구 22~03시 (구 단위)
- entertainment_cnt: 유흥시설 수 (구 단위)
- incon_report    : 불편신고 건수 (구 단위)
- infra_cnt       : 보안 인프라 - CCTV 수 (구 단위, 안전요인)
- safe_route_acc  : 안심귀갓길 시설물 수 (구 단위, 안전요인)
- GU_NM           : 자치구명
- risk_sc         : 위험 점수 (최종 위험지수 0~1)
- risk_lv         : 위험 등급 (낮음/보통/높음/매우높음)

계산 구조:
  crime_score   = (crime_data_norm + crime_rate_norm) / 2
  위험요인점수  = mean(crime_score, single_cnt, pop_tot, entertainment_cnt, incon_report)
  안전인프라점수 = mean(infra_cnt, safe_route_acc)  
  risk_sc       = minmax(위험요인점수 - 안전인프라점수)

생성 파일:
- core/01_gu_core.csv         : 구 단위 원 집계 데이터
- core/01_gu_scored.csv       : 정규화 + 위험지수 포함 데이터
"""

from pathlib import Path
import unicodedata
import zipfile
import pandas as pd
from IPython.display import display


# ════════════════════════════════════════
# 1. 기본 설정
# ════════════════════════════════════════
PROJECT_DIR = Path(r"C:\Users\njj09\OneDrive\바탕 화면\어딜가든")
CORE_DIR    = PROJECT_DIR
SAFETY_DIR  = PROJECT_DIR

OUTPUT_RAW    = CORE_DIR / "01_gu_core.csv"
OUTPUT_SCORED = CORE_DIR / "01_gu_scored.csv"

SEOUL_GU = [
    "종로구", "중구", "용산구", "성동구", "광진구",
    "동대문구", "중랑구", "성북구", "강북구", "도봉구",
    "노원구", "은평구", "서대문구", "마포구", "양천구",
    "강서구", "구로구", "금천구", "영등포구", "동작구",
    "관악구", "서초구", "강남구", "송파구", "강동구",
]

NIGHT_HOURS = [22, 23, 0, 1, 2, 3]

# 불편신고 기준 기간: 최근 3년
COMPLAINT_START = (2023, 3)
COMPLAINT_END   = (2026, 2)


def find_file_by_name(directory: Path, keyword: str, suffix: str | None = None) -> Path:
    """파일명에 특정 단어가 포함된 파일을 찾는다. macOS 한글 정규화 대응."""
    normalized_keyword = unicodedata.normalize("NFC", keyword)
    for path in directory.rglob("*"):
        normalized_name = unicodedata.normalize("NFC", path.name)
        if normalized_keyword in normalized_name and (suffix is None or path.suffix == suffix):
            return path
    raise FileNotFoundError(f"{directory} 안에서 '{keyword}' 파일을 찾지 못했습니다.")


FILES = {
    "single_cnt"      : find_file_by_name(PROJECT_DIR, "1인가구", ".zip"),
    "pop_tot"         : CORE_DIR / "서울특별시_250M격자_생활인구_내국인_utf8.csv",
    "crime_data"      : find_file_by_name(PROJECT_DIR, "5대범죄", ".csv"),
    "incon_report"    : CORE_DIR / "서울시 위치별 불편신고건수 정보.csv",
    "infra_cnt"       : find_file_by_name(SAFETY_DIR, "CCTV", ".xlsx"),
    "safe_route_acc"  : find_file_by_name(SAFETY_DIR, "안전시설물", ".csv"),
    "entertainment_cnt": CORE_DIR / "서울시_유흥단란주점_영업중.csv",
}


# ════════════════════════════════════════
# 2. 공통 함수
# ════════════════════════════════════════
def to_number(series: pd.Series) -> pd.Series:
    """쉼표가 포함된 문자열 숫자를 실수형으로 변환한다."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0)


def minmax(series: pd.Series) -> pd.Series:
    """0~1 범위로 정규화한다."""
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def print_file_paths() -> None:
    """사용 파일 목록을 출력한다."""
    print("사용 파일 목록")
    for name, path in FILES.items():
        print(f"  - {name}: {path}")


# ════════════════════════════════════════
# 3. 원자료별 전처리
# ════════════════════════════════════════
def read_single_cnt() -> pd.DataFrame:
    """1인가구 zip에서 구별 CSV를 합쳐 자치구별 single_cnt를 집계한다."""
    records = []
    with zipfile.ZipFile(FILES["single_cnt"]) as z:
        for name in z.namelist():
            with z.open(name) as f:
                records.append(pd.read_csv(f, encoding="cp949"))

    hh = pd.concat(records, ignore_index=True)
    hh["clean"]      = hh["행정구역"].str.replace(r"\(\d+\)", "", regex=True).str.strip()
    hh["GU_NM"]      = hh["clean"].str.extract(r"서울특별시\s+(\S+구)")
    hh["single_cnt"] = to_number(hh["2026년03월_거주자_총세대수"])

    result = (
        hh[hh["GU_NM"].notna()]
        .groupby("GU_NM", as_index=False)["single_cnt"]
        .sum()
    )

    return result[result["GU_NM"].isin(SEOUL_GU)].copy()

def read_pop_tot() -> pd.DataFrame:
    """생활인구에서 자치구별 야간(22~03시) pop_tot을 계산한다."""
    GU_CODE_MAP = {
    "11110": "종로구",  "11140": "중구",     "11170": "용산구",  "11200": "성동구",
    "11215": "광진구",  "11230": "동대문구",  "11260": "중랑구",  "11290": "성북구",
    "11305": "강북구",  "11320": "도봉구",   "11350": "노원구",  "11380": "은평구",
    "11410": "서대문구","11440": "마포구",   "11470": "양천구",  "11500": "강서구",
    "11530": "구로구",  "11545": "금천구",   "11560": "영등포구","11590": "동작구",
    "11620": "관악구",  "11650": "서초구",   "11680": "강남구",  "11710": "송파구",
    "11740": "강동구",
}

    usecols = ["시간", "행정동코드", "생활인구합계"]
    chunks  = pd.read_csv(FILES["pop_tot"], usecols=usecols, chunksize=200_000)
    results = []

    for chunk in chunks:
        night = chunk[chunk["시간"].isin(NIGHT_HOURS)].copy()
        night["생활인구합계"] = pd.to_numeric(night["생활인구합계"], errors="coerce").fillna(0)
        night["GU_CD"] = night["행정동코드"].astype(str).str[:5]
        results.append(night.groupby("GU_CD", as_index=False)["생활인구합계"].sum())

    pop = (
        pd.concat(results, ignore_index=True)
        .groupby("GU_CD", as_index=False)["생활인구합계"]
        .sum()
    )
    pop["GU_NM"] = pop["GU_CD"].map(GU_CODE_MAP)
    pop = pop.dropna(subset=["GU_NM"])

    return pop[["GU_NM", "생활인구합계"]].rename(columns={"생활인구합계": "pop_tot"})


def read_crime_data() -> pd.DataFrame:
    """5년평균 5대범죄 데이터에서 자치구별 crime_data를 집계한다."""
    df = pd.read_csv(FILES["crime_data"], encoding="utf-8", header=None)

    # 4행부터 실제 데이터 (0~3행은 헤더)
    # 컬럼1 = 자치구명, 컬럼2 = 발생건수 소계
    data = df.iloc[4:].copy()
    data = data[[1, 2]].copy()
    data.columns = ["GU_NM", "crime_data"]
    data["GU_NM"]     = data["GU_NM"].astype(str).str.strip()
    data["crime_data"] = to_number(data["crime_data"])

    return data[data["GU_NM"].isin(SEOUL_GU)].copy()


def read_incon_report() -> pd.DataFrame:
    """
    불편신고에서 최근 3년 자치구별 incon_report를 집계한다.

    원본 파일이 wide format (년도·월 행, 자치구 컬럼)이므로
    melt로 long format으로 변환 후 집계한다.
    """
    df = pd.read_csv(FILES["incon_report"], encoding="cp949")

    # wide → long format 변환
    df_long = df.melt(
        id_vars=["년도", "월"],
        value_vars=SEOUL_GU,
        var_name="GU_NM",
        value_name="incon_report",
    )
    df_long["incon_report"] = to_number(df_long["incon_report"])
    df_long = df_long.rename(columns={"년도": "연도"})

    # 최근 3년 필터링 (2023년 3월 ~ 2026년 2월)
    recent = df_long[
        (
            (df_long["연도"] > COMPLAINT_START[0]) &
            (df_long["연도"] < COMPLAINT_END[0])
        ) | (
            (df_long["연도"] == COMPLAINT_START[0]) &
            (df_long["월"] >= COMPLAINT_START[1])
        ) | (
            (df_long["연도"] == COMPLAINT_END[0]) &
            (df_long["월"] <= COMPLAINT_END[1])
        )
    ].copy()

    result = (
        recent
        .groupby("GU_NM", as_index=False)["incon_report"]
        .sum()
    )

    return result


def read_infra_cnt() -> pd.DataFrame:
    """자치구별 CCTV 수 infra_cnt를 읽는다."""
    df = pd.read_excel(FILES["infra_cnt"], header=2)

    df["GU_NM"] = (
        df["자치구"].astype(str).str.replace(" ", "", regex=False).str.strip()
    )
    df = df[df["GU_NM"].isin(SEOUL_GU)].copy()
    df["infra_cnt"] = to_number(df["총 계"])

    return df[["GU_NM", "infra_cnt"]]


def read_safe_route_acc() -> pd.DataFrame:
    """안심귀갓길 안전시설물 수 safe_route_acc를 자치구별로 집계한다."""
    df = pd.read_csv(
        r"C:\Users\njj09\OneDrive\바탕 화면\어딜가든\서울시 안심귀갓길 안전시설물.csv",
        encoding="cp949"
    )
    df["GU_NM"]          = df["시군구명"].str.extract(r"서울특별시\s+(\S+구)")
    df["safe_route_acc"] = to_number(df["설치대수"])
    result = df.groupby("GU_NM", as_index=False)["safe_route_acc"].sum()
    return result[result["GU_NM"].isin(SEOUL_GU)].copy()


def read_entertainment_cnt() -> pd.DataFrame:
    """유흥단란주점 영업 중 업소를 자치구별로 집계해 entertainment_cnt를 만든다."""
    df = pd.read_csv(FILES["entertainment_cnt"], encoding="utf-8")

    df["GU_NM"] = df["도로명주소"].str.extract(r"서울특별시\s+(\S+구)")

    result = (
        df[df["GU_NM"].isin(SEOUL_GU)]
        .groupby("GU_NM", as_index=False)
        .size()
        .rename(columns={"size": "entertainment_cnt"})
    )

    return result


# ════════════════════════════════════════
# 4. 데이터 통합 및 위험지수 계산
# ════════════════════════════════════════
def build_gu_core() -> pd.DataFrame:
    """자치구 단위 지표를 하나의 분석 테이블로 결합한다."""
    core = (
        read_crime_data()
        .merge(read_single_cnt(),        on="GU_NM", how="outer")
        .merge(read_incon_report(),       on="GU_NM", how="outer")
        .merge(read_pop_tot(),            on="GU_NM", how="outer")
        .merge(read_infra_cnt(),          on="GU_NM", how="outer")
        .merge(read_safe_route_acc(),     on="GU_NM", how="outer")
        .merge(read_entertainment_cnt(),  on="GU_NM", how="outer")
    )

    numeric_cols = [
        "crime_data", "single_cnt", "incon_report",
        "pop_tot", "infra_cnt", "safe_route_acc", "entertainment_cnt",
    ]

    core = core[core["GU_NM"].isin(SEOUL_GU)].copy()
    core[numeric_cols] = core[numeric_cols].fillna(0)

    # crime_rate: 야간생활인구 10만 명당 범죄 발생 건수
    # crime_data만으로는 인구 많은 구가 과대평가될 수 있어 상대 위험을 보정
    core["crime_rate"] = (
        core["crime_data"] / core["pop_tot"].replace(0, pd.NA)
    ) * 100_000
    core["crime_rate"] = core["crime_rate"].fillna(0)

    return core.sort_values("GU_NM").reset_index(drop=True)


def add_risk_score(core: pd.DataFrame) -> pd.DataFrame:
    """
    정규화 및 위험지수(risk_sc)를 계산한다.

    crime_score   = (crime_data_norm + crime_rate_norm) / 2
      → 범죄 발생 규모(crime_data)와 인구 대비 상대 위험(crime_rate)을 동시에 반영

    위험요인점수  = mean(crime_score, single_cnt, pop_tot, entertainment_cnt, incon_report)
    안전인프라점수 = mean(infra_cnt_norm_inv, safe_route_acc_norm_inv)
      → 인프라는 많을수록 안전하므로 역전(1 - norm) 적용

    risk_sc = minmax(위험요인점수 - 안전인프라점수)
    """
    scored = core.copy()

    # ── 범죄 지표 정규화 및 결합 ──
    scored["crime_data_norm"] = minmax(scored["crime_data"])
    scored["crime_rate_norm"] = minmax(scored["crime_rate"])
    scored["crime_score"]     = (scored["crime_data_norm"] + scored["crime_rate_norm"]) / 2

    # ── 위험 요인 정규화 ──
    for col in ["single_cnt", "pop_tot", "entertainment_cnt", "incon_report"]:
        scored[f"{col}_norm"] = minmax(scored[col])

    # ── 안전 인프라 정규화 (역전) ──
    for col in ["infra_cnt", "safe_route_acc"]:
        scored[f"{col}_norm"] = minmax(scored[col])  

    # ── 위험요인점수: 범죄점수 + 1인가구 + 야간인구 + 유흥시설 + 불편신고 ──
    scored["위험요인점수"] = scored[[
        "crime_score",
        "single_cnt_norm",
        "pop_tot_norm",
        "entertainment_cnt_norm",
        "incon_report_norm",
    ]].mean(axis=1)

    # ── 안전인프라점수: CCTV + 안심귀갓길 ──
    scored["안전인프라점수"] = scored[[
        "infra_cnt_norm",
        "safe_route_acc_norm",
    ]].mean(axis=1)

    # ── risk_sc: 위험요인 - 안전인프라 → 재정규화 ──
    scored["risk_sc_raw"] = scored["위험요인점수"] - scored["안전인프라점수"]
    scored["risk_sc"]     = minmax(scored["risk_sc_raw"])

    # ── risk_lv: 위험 등급 ──
    scored["risk_lv"] = pd.cut(
        scored["risk_sc"],
        bins=[0, 0.35, 0.50, 0.65, 1.0],
        labels=["낮음", "보통", "높음", "매우높음"],
        include_lowest=True,
    )

    return scored.sort_values("risk_sc", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════
# 5. 결과 출력
# ════════════════════════════════════════
def display_result(scored: pd.DataFrame) -> None:
    """위험지수 상위 10개 자치구의 계산 과정을 표로 출력한다."""
    cols = {
        "GU_NM"   : "자치구",
        "crime_score"  : "범죄 점수",
        "위험요인점수"  : "위험 요인 점수",
        "안전인프라점수": "안전 인프라 점수",
        "risk_sc_raw"  : "위험지수 원점수",
        "risk_sc"      : "risk_sc",
        "risk_lv"      : "risk_lv",
    }

    result = scored[list(cols.keys())].head(10).rename(columns=cols)
    numeric = ["범죄 점수", "위험 요인 점수", "안전 인프라 점수", "위험지수 원점수", "risk_sc"]
    result[numeric] = result[numeric].round(4)

    styled = (
        result.style
        .hide(axis="index")
        .background_gradient(subset=["risk_sc"], cmap="Reds")
        .format({c: "{:.4f}" for c in numeric})
    )
    display(styled)


# ════════════════════════════════════════
# 6. 메인
# ════════════════════════════════════════
def main() -> None:
    print_file_paths()

    core   = build_gu_core()
    scored = add_risk_score(core)

    core.to_csv(OUTPUT_RAW,    index=False, encoding="utf-8-sig")
    scored.to_csv(OUTPUT_SCORED, index=False, encoding="utf-8-sig")

    print("\n전처리 완료")
    print(f"  원 집계 데이터 : {OUTPUT_RAW}")
    print(f"  위험지수 데이터: {OUTPUT_SCORED}")
    print(f"  분석 자치구 수 : {len(core)}개")
    print("\n위험지수 상위 10개 자치구")
    display_result(scored)


main()
