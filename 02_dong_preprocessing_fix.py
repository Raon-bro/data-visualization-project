"""
서울시 야간 보행 안전 사각지대 분석 - 동 단위 데이터 전처리

변수명 규칙 (협업자 통일 규칙):
- single_cnt       : 1인가구 수 (동 단위)
- pop_tot          : 야간 생활인구 22~03시 (동 단위, 250M 격자 → 동 집계)
- entertainment_cnt: 유흥시설 수 (동 단위, 지번주소 → 동 추출)
- safe_route_acc   : 안심귀갓길 시설물 수 (동 단위, 안전요인)
- light_cnt        : 보안등 수 (동 단위, 좌표 → 동 공간조인, 안전요인)
                     ※ 동대문구는 동사무소 컬럼 기준 직접 집계
- transit_acc      : 심야 대중교통 접근성 (지하철역 수 + 심야버스 이용량, 안전요인)
- GU_NM            : 자치구명
- DONG_NM          : 행정동명
- DONG_CD          : 행정동코드
- risk_sc          : 잠재 위험 점수 (0~1)
- risk_lv          : 잠재 위험 등급

계산 구조:
  위험요인점수   = mean(single_cnt_norm, pop_tot_norm, entertainment_cnt_norm)
  안전인프라점수 = mean(safe_norm, light_norm, transit_norm)
  risk_sc = minmax(위험요인점수 - 안전인프라점수)

※ 동 단위 분석은 확보 가능한 변수만으로 산출한 "잠재적 위험 구역" 탐색용 분석임.
   범죄·CCTV 데이터는 구 단위만 존재해 포함하지 않음.

처리 원칙:
- 매핑 실패 동 중 인접 동 평균 대체: 강북구(번1·2·3동, 수유1·2·3동), 동대문구(신설동·용두동), 강남구(개포3동)
- 제외 동 (3개): 강동구 상일제1·2동(신설동), 구로구 항동(개발제한구역)
- 최종 분석 동 수: 424개

생성 파일:
- core/02_dong_core.csv    : 동 단위 원 집계 데이터
- core/02_dong_scored.csv  : 정규화 + 잠재위험지수 포함 데이터
"""

from pathlib import Path
import zipfile
import unicodedata
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from shapely import wkt

# ════════════════════════════════════════
# 1. 기본 설정
# ════════════════════════════════════════
PROJECT_DIR = Path(r"C:\Users\njj09\OneDrive\바탕 화면\어딜가든")

OUTPUT_RAW    = PROJECT_DIR / "02_dong_core.csv"
OUTPUT_SCORED = PROJECT_DIR / "02_dong_scored.csv"


EXCLUDE_DONG = {"상일제1동", "상일제2동", "항동"}
NIGHT_HOURS  = [22, 23, 0, 1, 2, 3]

NIGHT_BUS_COLS = [
    "22시승차총승객수", "23시승차총승객수", "00시승차총승객수",
    "1시승차총승객수",  "2시승차총승객수",  "3시승차총승객수",
    "22시하차총승객수", "23시하차총승객수", "00시하차총승객수",
    "1시하차총승객수",  "2시하차총승객수",  "3시하차총승객수",
]

plt.rcParams["font.family"]        = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def find_file_by_name(directory: Path, keyword: str, suffix: str | None = None) -> Path | None:
    """프로젝트 하위 폴더에서 파일명에 keyword가 들어간 파일을 찾는다."""
    keyword = unicodedata.normalize("NFC", keyword)
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        name = unicodedata.normalize("NFC", path.name)
        if keyword in name and (suffix is None or path.suffix == suffix):
            return path
    return None


FILES = {
    "single_cnt"       : find_file_by_name(PROJECT_DIR, "1인가구 구별 단위", ".zip"),
    "pop_tot"          : find_file_by_name(PROJECT_DIR, "250M격자_생활인구", ".csv"),
    "entertainment_cnt": find_file_by_name(PROJECT_DIR, "유흥단란주점", ".csv"),
    "safe_route_acc"   : find_file_by_name(PROJECT_DIR, "안심귀갓길 안전시설물", ".csv"),
    "light_cnt"        : find_file_by_name(PROJECT_DIR, "영등포구_보안등정보", ".zip"),
    "light_cnt_ddm"    : find_file_by_name(PROJECT_DIR, "동대문구_보안등정보", ".csv"),
    "subway"           : find_file_by_name(PROJECT_DIR, "역사마스터", ".csv"),
    "bus_time"         : find_file_by_name(PROJECT_DIR, "시간대별 승하차", ".csv"),
    "bus_route"        : find_file_by_name(PROJECT_DIR, "버스 노선 정보", ".csv"),
}


# ════════════════════════════════════════
# 2. 공통 함수
# ════════════════════════════════════════
def normalize_dong(name) -> str:
    """동 이름 표기 차이를 줄이기 위한 정규화."""
    return (
        str(name)
        .strip()
        .replace(" ", "")
        .replace("·", ".")
        .replace("ㆍ", ".")
        .replace("제", "")
    )


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


def read_csv_auto(path: Path, **kwargs) -> pd.DataFrame:
    """utf-8-sig, utf-8, cp949 순서로 CSV를 읽는다."""
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)

def print_file_paths() -> None:
    """사용 파일 목록을 출력한다."""
    print("사용 파일 목록")
    print(f"  - DONG_SHP: {find_file_by_name(PROJECT_DIR, 'bnd_dong_11_2025_2Q', '.shp')}")
    print(f"  - GU_SHP: {find_file_by_name(PROJECT_DIR, 'bnd_sigungu_11_2025_2Q', '.shp')}")
    for name, path in FILES.items():
        print(f"  - {name}: {path}")


# ════════════════════════════════════════
# 3. 경계 SHP 로드
# ════════════════════════════════════════
def load_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """동 경계와 구 경계를 로드하고, 동 경계에 GU_NM을 붙인다."""
    dong_shp = find_file_by_name(PROJECT_DIR, "bnd_dong_11_2025_2Q", ".shp")
    gu_shp   = find_file_by_name(PROJECT_DIR, "bnd_sigungu_11_2025_2Q", ".shp")

    dong = gpd.read_file(str(dong_shp)).to_crs(epsg=4326)
    gu   = gpd.read_file(str(gu_shp)).to_crs(epsg=4326)

    dong = dong.rename(columns={"ADM_CD": "DONG_CD", "ADM_NM": "DONG_NM"})
    gu   = gu.rename(columns={"SIGUNGU_CD": "GU_CD", "SIGUNGU_NM": "GU_NM"})

    dong["DONG_CD"]      = dong["DONG_CD"].astype(str).str[:8]
    dong["DONG_NM"]      = dong["DONG_NM"].astype(str).str.strip()
    dong["DONG_NM_norm"] = dong["DONG_NM"].apply(normalize_dong)

    # 동 대표점으로 구 이름 붙이기
    dong_point = dong[["DONG_CD", "DONG_NM", "DONG_NM_norm", "geometry"]].copy()
    dong_point["geometry"] = dong_point.geometry.representative_point()

    dong_gu = gpd.sjoin(
        dong_point,
        gu[["GU_NM", "geometry"]],
        how="left",
        predicate="within",
    )[["DONG_CD", "GU_NM"]]

    dong = dong.merge(dong_gu, on="DONG_CD", how="left")

    return dong, gu

def spatial_join_to_dong(gdf: gpd.GeoDataFrame, dong: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """좌표 GeoDataFrame을 동 경계와 공간조인한다.
    경계 밖 좌표는 가장 가까운 동에 매핑하고, 서울 외 데이터는 GU_NM으로 필터링."""
    joined = gpd.sjoin_nearest(
        gdf,
        dong[["DONG_CD", "GU_NM", "DONG_NM", "DONG_NM_norm", "geometry"]],
        how="left",
    )
    return joined


# ════════════════════════════════════════
# 4. 변수별 전처리
# ════════════════════════════════════════
def make_dong_base(dong: gpd.GeoDataFrame) -> pd.DataFrame:
    """동 경계 SHP 기준으로 분석 기준 동 목록을 만든다."""
    base = dong[["DONG_CD", "GU_NM", "DONG_NM", "DONG_NM_norm"]].drop_duplicates()
    base = base[~base["DONG_NM"].isin(EXCLUDE_DONG)].copy()
    return base.sort_values(["GU_NM", "DONG_NM"]).reset_index(drop=True)


def read_single_cnt() -> pd.DataFrame:
    """1인가구 데이터에서 동 단위 single_cnt를 추출한다."""
    path = FILES["single_cnt"]

    if path.suffix == ".zip":
        records = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    records.append(pd.read_csv(f, encoding="cp949"))
        hh = pd.concat(records, ignore_index=True)
        hh["clean"]   = hh["행정구역"].str.replace(r"\(\d+\)", "", regex=True).str.strip()
        hh["DONG_CD"] = hh["행정구역"].str.extract(r"\((\d+)\)").iloc[:, 0].str[:8]
        hh["GU_NM"]   = hh["clean"].str.extract(r"서울특별시\s+(\S+구)")
        hh["DONG_NM"] = hh["clean"].str.extract(r"\S+구\s+(.+)$")
    else:
        hh = read_csv_auto(path)
        hh = hh[hh["row_type"] == "dong"].copy()
        hh["GU_NM"]   = hh["district_name"].astype(str).str.strip()
        hh["DONG_NM"] = hh["admin_name"].astype(str).str.strip()
        hh["DONG_CD"] = hh["행정구역"].str.extract(r"\((\d+)\)").iloc[:, 0].str[:8]

    hh["DONG_NM_norm"] = hh["DONG_NM"].apply(normalize_dong)
    hh["single_cnt"]   = to_number(hh["2026년03월_거주자_총세대수"])

    return (
        hh[["DONG_CD", "GU_NM", "DONG_NM", "DONG_NM_norm", "single_cnt"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def read_pop_tot() -> pd.DataFrame:
    """
    생활인구에서 야간(22~03시) 동별 pop_tot을 집계한다.
    행정동코드 → 동 이름 → SHP DONG_CD 순으로 매핑한다.
    """
    # ── 1. 생활인구 야간 집계 ──
    chunks = pd.read_csv(
        FILES["pop_tot"],
        usecols=["시간", "행정동코드", "생활인구합계"],
        chunksize=200_000,
    )
    results = []
    for chunk in chunks:
        night = chunk[chunk["시간"].isin(NIGHT_HOURS)].copy()
        night["생활인구합계"] = pd.to_numeric(night["생활인구합계"], errors="coerce").fillna(0)
        results.append(night.groupby("행정동코드", as_index=False)["생활인구합계"].sum())

    pop = (
        pd.concat(results, ignore_index=True)
        .groupby("행정동코드", as_index=False)["생활인구합계"]
        .sum()
        .rename(columns={"행정동코드": "HH_CD", "생활인구합계": "pop_tot"})
    )
    pop["HH_CD"] = pop["HH_CD"].astype(str).str[:8]

    # ── 2. 1인가구 파일로 행정동코드 → 동 이름 매핑 테이블 만들기 ──
    records = []
    with zipfile.ZipFile(FILES["single_cnt"]) as z:
        for name in z.namelist():
            with z.open(name) as f:
                records.append(pd.read_csv(f, encoding="cp949"))

    hh = pd.concat(records, ignore_index=True)
    hh["clean"]   = hh["행정구역"].str.replace(r"\(\d+\)", "", regex=True).str.strip()
    hh["HH_CD"]   = hh["행정구역"].str.extract(r"\((\d+)\)").iloc[:, 0].str[:8]
    hh["GU_NM"]   = hh["clean"].str.extract(r"서울특별시\s+(\S+구)")
    hh["DONG_NM"] = hh["clean"].str.extract(r"\S+구\s+(.+)$")
    hh["DONG_NM_norm"] = hh["DONG_NM"].apply(normalize_dong)

    hh_map = (
        hh[hh["GU_NM"].notna()]
        [["HH_CD", "GU_NM", "DONG_NM_norm"]]
        .drop_duplicates()
        .dropna()
    )

    # ── 3. 동 경계 SHP의 DONG_CD 매핑 ──
    dong_base, _ = load_boundaries()
    dong_cd_map = (
        dong_base[["DONG_CD", "DONG_NM_norm"]]
        .drop_duplicates()
        .set_index("DONG_NM_norm")["DONG_CD"]
        .to_dict()
    )

    hh_map["DONG_CD"] = hh_map["DONG_NM_norm"].map(dong_cd_map)

    # ── 4. pop_tot + HH_CD → DONG_CD 연결 ──
    pop = pop.merge(hh_map[["HH_CD", "DONG_CD"]], on="HH_CD", how="left")
    pop = pop[pop["DONG_CD"].notna()]

    result = pop.groupby("DONG_CD", as_index=False)["pop_tot"].sum()
    print(f"  pop_tot 매핑 성공: {len(result)}개 동")
    return result

def read_entertainment_cnt() -> pd.DataFrame:
    """
    유흥단란주점 TM 좌표를 WGS84로 변환 후 동 경계와 공간조인해
    동별 entertainment_cnt를 집계한다.
    원본 좌표가 EPSG:5174(TM) 체계라 변환 필요.
    """
    if FILES["entertainment_cnt"] is None:
        return pd.DataFrame(columns=["DONG_CD", "entertainment_cnt"])

    df = read_csv_auto(FILES["entertainment_cnt"])
    df["X"] = pd.to_numeric(df["좌표정보(X)"], errors="coerce")
    df["Y"] = pd.to_numeric(df["좌표정보(Y)"], errors="coerce")
    df = df.dropna(subset=["X", "Y"])

    # TM(EPSG:5174) > WGS84(EPSG:4326) 변환
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["X"], df["Y"]), crs="EPSG:5174"
    )
    gdf = gdf.to_crs(epsg=4326)

    dong_base = load_boundaries()[0]  # 동 경계 SHP
    joined = gpd.sjoin_nearest(
        gdf, dong_base[["DONG_CD", "DONG_NM", "geometry"]], how="left"
    )

    return (
        joined[joined["DONG_CD"].notna()]
        .groupby("DONG_CD", as_index=False)
        .size()
        .rename(columns={"size": "entertainment_cnt"})
    )


def read_safe_route_acc() -> pd.DataFrame:
    """안심귀갓길 시설물 데이터에서 동별 safe_route_acc를 집계한다."""
    df = pd.read_csv(FILES["safe_route_acc"], encoding="cp949")
    df["geometry"] = df["포인트 wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    dong_base = load_boundaries()[0]
    joined = gpd.sjoin(gdf, dong_base[["DONG_CD","DONG_NM","geometry"]], 
                       how="left", predicate="within")
    result = (joined[joined["DONG_CD"].notna()]
              .groupby("DONG_CD", as_index=False)["설치대수"].sum()
              .rename(columns={"설치대수": "safe_route_acc"}))
    return result



def read_light_cnt(dong_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """보안등 zip에서 25개 구 전체를 읽어 동별 light_cnt를 집계한다."""
    print("💡 보안등 데이터 좌표 분석 및 공간 조인 시작...")

    light_path = FILES.get("light_cnt")
    if not light_path or not os.path.exists(light_path):
        print("⚠️ 보안등 파일을 찾을 수 없습니다.")
        return pd.DataFrame(columns=["DONG_CD", "light_cnt"])

    results = []

    with zipfile.ZipFile(light_path) as z:
        for fname in z.namelist():
            df = None
            for enc in ["utf-8", "utf-8-sig", "cp949"]:
                try:
                    with z.open(fname) as f:
                        df = pd.read_csv(f, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue

            if df is None or "위도" not in df.columns or "경도" not in df.columns:
                continue

            df["위도"] = pd.to_numeric(df["위도"], errors="coerce")
            df["경도"] = pd.to_numeric(df["경도"], errors="coerce")
            df = df.dropna(subset=["위도", "경도"])
            df["설치개수"] = pd.to_numeric(df["설치개수"], errors="coerce").fillna(1)

            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df["경도"], df["위도"]), crs="EPSG:4326"
            )
            joined = gpd.sjoin_nearest(gdf, dong_gdf[["DONG_CD", "DONG_NM", "geometry"]], how="left")
            joined = joined[joined["DONG_CD"].notna()]

            agg = (
                joined.groupby("DONG_CD", as_index=False)["설치개수"]
                .sum()
                .rename(columns={"설치개수": "light_cnt"})
            )
            results.append(agg)

    if not results:
        return pd.DataFrame(columns=["DONG_CD", "light_cnt"])

    light_df = (
        pd.concat(results, ignore_index=True)
        .groupby("DONG_CD", as_index=False)["light_cnt"]
        .sum()
    )
    print(f"✅ 총 {len(light_df)}개 행정동의 보안등 집계 완료")
    return light_df
   


def read_transit_acc(dong: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    지하철역 수와 심야버스 야간 이용량을 동별로 집계해 transit_acc를 만든다.
    transit_acc = subway_cnt + bus_usage_norm
    """
    if FILES["subway"] is None or FILES["bus_time"] is None or FILES["bus_route"] is None:
        print("  대중교통 파일 없음 → transit_acc 제외")
        return pd.DataFrame(columns=["DONG_CD", "transit_acc"])

    SEOUL_GU = [
        "종로구", "중구", "용산구", "성동구", "광진구",
        "동대문구", "중랑구", "성북구", "강북구", "도봉구",
        "노원구", "은평구", "서대문구", "마포구", "양천구",
        "강서구", "구로구", "금천구", "영등포구", "동작구",
        "관악구", "서초구", "강남구", "송파구", "강동구",
    ]

    # 지하철
    subway     = read_csv_auto(FILES["subway"])
    subway_gdf = gpd.GeoDataFrame(
        subway,
        geometry=gpd.points_from_xy(subway["경도"], subway["위도"]),
        crs="EPSG:4326",
    )
    subway_joined = spatial_join_to_dong(subway_gdf, dong)
    subway_cnt = (
        subway_joined[subway_joined["GU_NM"].isin(SEOUL_GU)]
        .groupby("DONG_CD", as_index=False)
        .size()
        .rename(columns={"size": "subway_cnt"})
    )

    # 심야버스
    bus_time  = read_csv_auto(FILES["bus_time"], low_memory=False)
    night_bus = bus_time[bus_time["교통수단타입명"] == "서울심야버스"].copy()
    night_bus["심야이용합계"] = night_bus[NIGHT_BUS_COLS].sum(axis=1)

    stop_usage = (
        night_bus.groupby("버스정류장ARS번호")["심야이용합계"]
        .sum()
        .reset_index()
    )
    stop_usage.columns   = ["ARS_ID", "심야이용합계"]
    stop_usage["ARS_ID"] = stop_usage["ARS_ID"].astype(str)

    bus_route           = read_csv_auto(FILES["bus_route"])
    bus_route["ARS_ID"] = bus_route["ARS_ID"].astype(str)

    stop_coord = stop_usage.merge(
        bus_route[["ARS_ID", "X좌표", "Y좌표"]].drop_duplicates("ARS_ID"),
        on="ARS_ID", how="left",
    ).dropna(subset=["X좌표", "Y좌표"])

    bus_gdf = gpd.GeoDataFrame(
        stop_coord,
        geometry=gpd.points_from_xy(stop_coord["X좌표"], stop_coord["Y좌표"]),
        crs="EPSG:4326",
    )
    bus_joined = spatial_join_to_dong(bus_gdf, dong)
    bus_usage  = (
        bus_joined[bus_joined["DONG_CD"].notna()]
        .groupby("DONG_CD", as_index=False)["심야이용합계"]
        .sum()
        .rename(columns={"심야이용합계": "bus_usage"})
    )

    transit = subway_cnt.merge(bus_usage, on="DONG_CD", how="outer").fillna(0)
    transit["bus_usage_norm"] = minmax(transit["bus_usage"])
    transit["transit_acc"]    = transit["subway_cnt"] + transit["bus_usage_norm"]

    return transit[["DONG_CD", "transit_acc"]]

# ════════════════════════════════════════
# 5. 데이터 통합
# ════════════════════════════════════════
def build_dong_core(dong: gpd.GeoDataFrame) -> pd.DataFrame:
    """동 SHP 기준으로 동 단위 변수를 하나의 분석 테이블로 결합한다."""
    print("\n데이터 로딩 중...")

    base    = make_dong_base(dong)
    single  = read_single_cnt()
    pop     = read_pop_tot()
    ent     = read_entertainment_cnt()
    safe    = read_safe_route_acc()

    print("\n보안등 집계 중...")
    light   = read_light_cnt(dong)

    print("\n대중교통 집계 중...")
    transit = read_transit_acc(dong)

    # DONG_CD 기준 병합 (single은 코드/이름 두 방식으로 시도)
    single_by_cd = single.groupby("DONG_CD", as_index=False)["single_cnt"].sum()
    single_by_nm = (
        single.groupby(["GU_NM", "DONG_NM_norm"], as_index=False)["single_cnt"]
        .sum()
        .rename(columns={"single_cnt": "single_cnt_name"})
    )

    core = (
        base
        .merge(single_by_cd, on="DONG_CD", how="left")
        .merge(single_by_nm, on=["GU_NM", "DONG_NM_norm"], how="left")
    )
    core["single_cnt"] = core["single_cnt"].fillna(core["single_cnt_name"])
    core = core.drop(columns=["single_cnt_name"])

    core = (
        core
        .merge(pop,     on="DONG_CD",                how="left")
        .merge(light,   on="DONG_CD",                how="left")
        .merge(transit, on="DONG_CD",                how="left")
        .merge(ent, on="DONG_CD", how="left")
        .merge(safe, on="DONG_CD", how="left")
    )

    numeric_cols = [
        "single_cnt", "pop_tot", "entertainment_cnt",
        "safe_route_acc", "light_cnt", "transit_acc",
    ]
    for col in numeric_cols:
        if col not in core.columns:
            core[col] = 0

    core[numeric_cols] = core[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # pop_tot = 0인 동 → 구 평균 대체
    zero_pop = core["pop_tot"] == 0
    if zero_pop.sum() > 0:
        gu_mean = core[~zero_pop].groupby("GU_NM")["pop_tot"].mean()
        core.loc[zero_pop, "pop_tot"] = core.loc[zero_pop, "GU_NM"].map(gu_mean)
        core["pop_tot"] = core["pop_tot"].fillna(0)
        print(f"\npop_tot 구 평균 대체: {zero_pop.sum()}개 동")

    print(f"\n보안등 집계 확인")
    print(f"  전체 합계: {core['light_cnt'].sum():,.0f}개")
    print(f"  1개 이상 동 수: {(core['light_cnt'] > 0).sum():,}개")
    print(f"  0개 동 수: {(core['light_cnt'] == 0).sum():,}개")

    return core.sort_values(["GU_NM", "DONG_NM"]).reset_index(drop=True)


# ════════════════════════════════════════
# 6. 잠재 위험지수 계산
# ════════════════════════════════════════
def add_risk_score(core: pd.DataFrame) -> pd.DataFrame:
    """
    동 단위 잠재 위험지수 risk_sc를 계산한다.

    위험요인점수   = mean(single_cnt_norm, pop_tot_norm, entertainment_cnt_norm)
    안전인프라점수 = mean(safe_norm, light_norm, transit_norm)
    risk_sc = minmax(위험요인점수 - 안전인프라점수)

    ※ 안전 변수는 많을수록 안전하므로 역전(1 - norm) 적용
    ※ transit_acc가 전부 0이면 안전인프라에서 제외
    """
    scored = core.copy()

    risk_cols  = ["single_cnt", "pop_tot", "entertainment_cnt"]
    infra_cols = ["safe_route_acc", "light_cnt"]

    if "transit_acc" in scored.columns and scored["transit_acc"].sum() > 0:
        infra_cols.append("transit_acc")

    for col in risk_cols:
        scored[f"{col}_norm"] = minmax(scored[col])

    for col in infra_cols:
        scored[f"{col}_norm"] = minmax(scored[col])  

    scored["위험요인점수"]   = scored[[f"{col}_norm" for col in risk_cols]].mean(axis=1)
    scored["안전인프라점수"] = scored[[f"{col}_norm" for col in infra_cols]].mean(axis=1)

    scored["risk_sc_raw"] = scored["위험요인점수"] - scored["안전인프라점수"]
    scored["risk_sc"]     = minmax(scored["risk_sc_raw"])

    # 4분위수 기반 등급 (임의 구간 사용 안 함)
    scored["risk_lv"] = pd.qcut(
        scored["risk_sc"],
        q=4,
        labels=["낮음", "보통", "높음", "매우높음"],
        duplicates="drop",
    )

    return scored.sort_values("risk_sc", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════
# 7. 결과 출력
# ════════════════════════════════════════
def display_summary_cards(core: pd.DataFrame, scored: pd.DataFrame) -> None:
    """전처리 결과 요약을 카드 형태로 출력한다."""
    html = f"""
    <div style="
        display:grid; grid-template-columns:repeat(4,1fr);
        gap:12px; margin:12px 0 18px 0; font-family:AppleGothic,sans-serif;
    ">
        <div style="padding:14px;border-radius:10px;background:#F8F9FA;border:1px solid #DDD;">
            <div style="font-size:13px;color:#777;">분석 동 수</div>
            <div style="font-size:24px;font-weight:700;">{len(core):,}개</div>
        </div>
        <div style="padding:14px;border-radius:10px;background:#FFF5F5;border:1px solid #F5C2C7;">
            <div style="font-size:13px;color:#777;">최고 위험 동</div>
            <div style="font-size:20px;font-weight:700;">
                {scored.iloc[0]["GU_NM"]} {scored.iloc[0]["DONG_NM"]}
            </div>
        </div>
        <div style="padding:14px;border-radius:10px;background:#F8F9FA;border:1px solid #DDD;">
            <div style="font-size:13px;color:#777;">최고 risk_sc</div>
            <div style="font-size:24px;font-weight:700;">{scored["risk_sc"].max():.4f}</div>
        </div>
        <div style="padding:14px;border-radius:10px;background:#F8F9FA;border:1px solid #DDD;">
            <div style="font-size:13px;color:#777;">평균 risk_sc</div>
            <div style="font-size:24px;font-weight:700;">{scored["risk_sc"].mean():.4f}</div>
        </div>
    </div>
    """
    display(HTML(html))


def display_pretty_dong_result(scored: pd.DataFrame, top_n: int = 20) -> None:
    """동 단위 잠재위험지수 결과를 표와 그래프로 출력한다."""
    show_cols = [
        "GU_NM", "DONG_NM", "single_cnt", "pop_tot", "entertainment_cnt",
        "safe_route_acc", "light_cnt", "위험요인점수", "안전인프라점수", "risk_sc", "risk_lv",
    ]
    col_rename = {
        "GU_NM": "구", "DONG_NM": "동", "single_cnt": "1인가구",
        "pop_tot": "야간생활인구", "entertainment_cnt": "유흥시설",
        "safe_route_acc": "안심귀갓길", "light_cnt": "보안등",
        "위험요인점수": "위험요인", "안전인프라점수": "안전인프라",
        "risk_sc": "잠재위험지수", "risk_lv": "등급",
    }

    result  = scored[show_cols].head(top_n).copy().rename(columns=col_rename)
    num_fmt = {
        "1인가구": "{:,.0f}", "야간생활인구": "{:,.0f}",
        "유흥시설": "{:,.0f}", "안심귀갓길": "{:,.0f}", "보안등": "{:,.0f}",
        "위험요인": "{:.4f}", "안전인프라": "{:.4f}", "잠재위험지수": "{:.4f}",
    }

    styled = (
        result.style
        .hide(axis="index")
        .background_gradient(subset=["잠재위험지수"], cmap="Reds")
        .format(num_fmt)
    )

    display(HTML(f"<h3>잠재 위험 상위 {top_n}개 동</h3>"))
    display(styled)

    plot_df       = result.sort_values("잠재위험지수", ascending=True).copy()
    plot_df["지역"] = plot_df["구"] + " " + plot_df["동"]

    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    bars = plt.barh(plot_df["지역"], plot_df["잠재위험지수"], color="#C0392B", alpha=0.82)

    for bar, value in zip(bars, plot_df["잠재위험지수"]):
        plt.text(
            value + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}", va="center", fontsize=9,
        )

    plt.title(f"잠재 위험 상위 {top_n}개 동", fontsize=14, fontweight="bold")
    plt.xlabel("잠재위험지수 risk_sc")
    plt.xlim(0, 1.08)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════
# 8. 메인
# ════════════════════════════════════════
def main() -> None:
    print_file_paths()

    print("\n경계 SHP 로딩...")
    dong, gu = load_boundaries()
    print(f"  동 경계: {len(dong)}개")
    print(f"  구 경계: {len(gu)}개")

    core   = build_dong_core(dong)
    scored = add_risk_score(core)

    core.to_csv(OUTPUT_RAW,    index=False, encoding="utf-8-sig")
    scored.to_csv(OUTPUT_SCORED, index=False, encoding="utf-8-sig")

    print(f"\n전처리 완료")
    print(f"  원 집계 데이터 : {OUTPUT_RAW}")
    print(f"  잠재위험지수 데이터: {OUTPUT_SCORED}")
    print(f"  분석 동 수: {len(core)}개")

    display_summary_cards(core, scored)
    display_pretty_dong_result(scored, top_n=20)

main()
print_file_paths()