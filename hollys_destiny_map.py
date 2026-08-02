"""
할리스 매장 크롤링 + 카카오 지오코딩 + 인구 대비 밀도 분석 + 지도 시각화

원본 노트북(data16_hollys_kakao_map.ipynb)을 단계별 함수로 정리한 스크립트.
각 단계는 이전 단계가 만든 source/*.csv 를 그대로 읽어서 쓰기 때문에,
매번 전체를 처음부터 돌릴 필요 없이 필요한 단계만 골라서 실행할 수 있다.

실행 예)
    uv run python hollys_kakao_map.py --step crawl
    uv run python hollys_kakao_map.py --step geocode
    uv run python hollys_kakao_map.py --step all

사전 준비
    - .env 파일에 KAKAO_API_KEY=xxxx 저장 (2단계 지오코딩에 필요)
    - source/ 폴더에 아래 원본 파일 존재
        - 행정구역_시군구_별__성별_인구수_20260731154430.csv (KOSIS 인구 데이터)
        - skorea-provinces-2018-geo.json (시도 경계 GeoJSON)
    - output/ 폴더 (없으면 자동 생성)
"""

import argparse
import datetime
import json
import os
import platform
import re
import time

import folium
import numpy as np
import pandas as pd
import pytz
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm

SOURCE_DIR = "source"
OUTPUT_DIR = "output"

BASE_URL = "https://www.hollys.co.kr/store/korea/korStore2.do"

# 시도명 표준화 매핑 (2, 3단계 공통 사용)
SIDO_MAP = {
    "서울": "서울특별시", "서울시": "서울특별시", "서울특별시": "서울특별시",
    "부산": "부산광역시", "부산시": "부산광역시", "부산광역시": "부산광역시",
    "대구": "대구광역시", "대구시": "대구광역시", "대구광역시": "대구광역시",
    "인천": "인천광역시", "인천시": "인천광역시", "인천광역시": "인천광역시",
    "광주": "광주광역시", "광주시": "광주광역시", "광주광역시": "광주광역시",
    "대전": "대전광역시", "대전시": "대전광역시", "대전광역시": "대전광역시",
    "울산": "울산광역시", "울산시": "울산광역시", "울산광역시": "울산광역시",
    "세종": "세종특별자치시", "세종시": "세종특별자치시", "세종특별자치시": "세종특별자치시",
    "경기": "경기도", "경기도": "경기도",
    "강원": "강원특별자치도", "강원도": "강원특별자치도", "강원특별자치도": "강원특별자치도",
    "충북": "충청북도", "충청북도": "충청북도",
    "충남": "충청남도", "충청남도": "충청남도",
    "전북": "전북특별자치도", "전라북도": "전북특별자치도", "전북특별자치도": "전북특별자치도",
    "전남": "전라남도", "전라남도": "전라남도",
    "경북": "경상북도", "경상북도": "경상북도",
    "경남": "경상남도", "경상남도": "경상남도",
    "제주": "제주특별자치도", "제주도": "제주특별자치도", "제주특별자치도": "제주특별자치도",
}


def setup_env():
    """폴더 준비 + OS별 한글 폰트 설정 (Mac은 AppleGothic)"""
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if platform.system() == "Windows":
        plt.rc("font", family="Malgun Gothic")
    elif platform.system() == "Darwin":
        plt.rc("font", family="AppleGothic")
    else:
        plt.rc("font", family="NanumBarunGothic")

    plt.rc("axes", unicode_minus=False)


# =====================================================
# 1단계: 할리스 매장 크롤링
# =====================================================
def _parse_paging_info(soup):
    """페이지네이션에서 페이지번호 목록 + 다음 블록 페이지 번호를 파싱"""
    paging_div = soup.select_one("div.paging")
    if paging_div is None:
        return [], None

    page_numbers = []
    for tag in paging_div.select("a, strong"):
        txt = tag.get_text(strip=True)
        if txt.isdigit():
            page_numbers.append(int(txt))

    next_block_page = None
    for a in paging_div.select("a[onclick]"):
        onclick_text = a.get("onclick")
        match = re.search(r"paging\((\d+)\s*,\s*1\)", onclick_text)
        if match:
            next_block_page = int(match.group(1))
            break

    return page_numbers, next_block_page


def _get_total_pages():
    """블록 이동하면서 전체 페이지 수를 끝까지 탐색"""
    page = 1
    max_page = 1

    while True:
        print(f"총페이지 탐색중... (현재 확인 페이지: {page})")
        res = requests.get(BASE_URL, params={"pageNo": page})
        soup = BeautifulSoup(res.text, "html.parser")

        page_numbers, next_block_page = _parse_paging_info(soup)
        if page_numbers:
            max_page = max(max_page, max(page_numbers))

        if next_block_page is None:
            break

        page = next_block_page
        time.sleep(0.2)

    print("최종 확인된 총 페이지 수:", max_page)
    return max_page


def _crawl_store_page(page):
    """특정 페이지의 매장 데이터 크롤링 (매장서비스 아이콘 alt 텍스트 포함)"""
    res = requests.get(BASE_URL, params={"pageNo": page})
    if res.status_code != 200:
        print(f"{page}페이지 요청 실패:", res.status_code)
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    tbody = soup.select_one("table.tb_store tbody")
    if tbody is None:
        return []

    page_result = []
    for row in tbody.select("tr"):
        tds = row.select("td")
        if len(tds) < 6:  # Hollys 테이블은 td 6개 구조
            continue

        area = tds[0].get_text(strip=True)
        name = tds[1].get_text(strip=True)
        status = tds[2].get_text(strip=True)
        addr = tds[3].get_text(strip=True)

        service_td = tds[4]
        service_list = [img.get("alt").strip() for img in service_td.select("img") if img.get("alt")]
        store_service = "/".join(service_list)

        phone = tds[5].get_text(strip=True)
        page_result.append([area, name, status, addr, store_service, phone])

    return page_result


def step1_crawl_stores():
    """할리스 홈페이지 매장검색 페이지를 전부 크롤링해서 source/hollys_store.csv 저장"""
    total_pages = _get_total_pages()

    all_data = []
    for page in range(1, total_pages + 1):
        print(f"매장 수집중: {page}/{total_pages}")
        all_data.extend(_crawl_store_page(page))
        time.sleep(0.3)

    df = pd.DataFrame(all_data, columns=["지역", "매장명", "현황", "주소", "매장서비스", "전화번호"])
    print("\n최종 매장 수:", len(df))
    print(df.head())

    # 참고용: 수집 시각 (파일명에는 미사용, 필요하면 KST 타임스탬프로 활용)
    to_now = datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
    print("수집 시각:", to_now)

    out_path = f"{SOURCE_DIR}/hollys_store.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("저장 완료:", out_path)
    return df


# =====================================================
# 2단계: 주소 → 위도/경도 변환 (카카오 지오코딩)
# =====================================================
def _clean_address(address):
    """지오코딩 성공률을 높이기 위한 주소 전처리 (괄호/층/호수/지하 등 제거)"""
    if pd.isna(address):
        return ""

    addr = str(address)
    addr = re.sub(r"\(.*?\)", "", addr)  # 괄호 내용 제거
    addr = addr.split(",")[0]  # 쉼표 뒤 제거

    remove_patterns = [
        r"\d+\s*층", r"\d+\s*호", r"지하\s*\d*",
        r"B\d+", r"\d+F", r"\d+~\d+층", r"\d+~\d+", r"\s*층",
    ]
    for pattern in remove_patterns:
        addr = re.sub(pattern, "", addr)

    addr = addr.replace("·", " ")
    addr = re.sub(r"\s+", " ", addr)
    return addr.strip()


def _kakao_address_search(query, api_key):
    """카카오 주소검색 API: 정식 주소 -> 좌표"""
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    response = requests.get(url, headers=headers, params={"query": query})

    if response.status_code != 200:
        print("주소검색 요청 실패:", response.status_code, response.text)
        return None, None

    result = response.json()
    if result["documents"]:
        x = result["documents"][0]["x"]  # 경도
        y = result["documents"][0]["y"]  # 위도
        return float(y), float(x)
    return None, None


def _kakao_keyword_search(query, api_key):
    """카카오 키워드검색 API: 주소검색 실패 시 대체 (휴게소 등 비정형 주소용)"""
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    response = requests.get(url, headers=headers, params={"query": query})

    if response.status_code != 200:
        print("키워드검색 요청 실패:", response.status_code, response.text)
        return None, None

    result = response.json()
    if result["documents"]:
        x = result["documents"][0]["x"]
        y = result["documents"][0]["y"]
        return float(y), float(x)
    return None, None


def _extract_rest_area(store_name):
    """휴게소점 전용: '까치골휴게소(상)점' -> '까치골휴게소' 처럼 키워드검색용 이름 추출"""
    rest_name = store_name.replace("(상)", "").replace("(하)", "")
    rest_name = rest_name.replace("휴게소점", "휴게소")
    return rest_name.strip()


def step2_geocode():
    """
    source/hollys_store.csv 의 매장 주소를 카카오 API로 위도/경도 변환.
    1차 주소검색 실패 시 2차로 '매장명(+휴게소면 지명) + 할리스' 키워드검색을 시도.
    결과에 시도(광역시/도) 컬럼을 표준화해 추가하고
    source/hollys_store_geo_kakao_final.csv 로 저장.
    """
    load_dotenv()
    api_key = os.getenv("KAKAO_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 KAKAO_API_KEY가 설정되어 있지 않습니다.")

    df = pd.read_csv(f"{SOURCE_DIR}/hollys_store.csv")

    lat_list, lon_list, clean_addr_list, method_list = [], [], [], []

    for store, addr in tqdm(zip(df["매장명"], df["주소"]), total=len(df)):
        cleaned_addr = _clean_address(addr)
        clean_addr_list.append(cleaned_addr)

        # 1차: 주소검색
        lat, lon = _kakao_address_search(cleaned_addr, api_key)
        if lat is not None:
            lat_list.append(lat)
            lon_list.append(lon)
            method_list.append("주소검색")
            time.sleep(0.2)
            continue

        # 2차: 키워드검색 (휴게소점이면 휴게소명으로)
        if "휴게소" in store:
            keyword = _extract_rest_area(store) + " 할리스"
        else:
            keyword = store + " 할리스"

        lat, lon = _kakao_keyword_search(keyword, api_key)
        if lat is not None:
            lat_list.append(lat)
            lon_list.append(lon)
            method_list.append("키워드검색")
        else:
            lat_list.append(None)
            lon_list.append(None)
            method_list.append("실패")

        time.sleep(0.2)

    df["주소_전처리"] = clean_addr_list
    df["위도"] = lat_list
    df["경도"] = lon_list
    df["검색방식"] = method_list

    print(df.head(10))
    print("좌표 변환 성공률:", df["위도"].notnull().mean())

    if "시도" not in df.columns:
        df["시도"] = df["주소"].astype(str).str.split().str[0]
    df["시도"] = df["시도"].replace(SIDO_MAP)

    out_path = f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print("저장 완료:", out_path)
    return df


# =====================================================
# 3단계: 시도별 매장 수 계산
# =====================================================
def step3_count_by_sido():
    """시도별 매장 수 집계 (DataFrame만 반환, 별도 저장 없음 -> 5단계에서 사용)"""
    df_store = pd.read_csv(f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv")
    store_count = df_store["시도"].value_counts().reset_index()
    store_count.columns = ["시도", "매장수"]
    print(store_count)
    return store_count


# =====================================================
# 4단계: 인구 데이터 정리 (population_sido.csv 생성)
# =====================================================
def step4_prepare_population():
    """
    KOSIS 원본 인구 CSV에서 불필요한 행/컬럼을 정리하고
    '시도' / '인구' 컬럼만 남겨 source/population_sido.csv로 저장.
    """
    df = pd.read_csv(
        f"{SOURCE_DIR}/행정구역_시군구_별__성별_인구수_20260731154430.csv",
        encoding="utf-8",
    )

    # 1) 헤더/전국 합계 행 제거
    df = df[~df["행정구역(시군구)별"].str.contains(r"행정구역\(시군구\)별|전국", na=False)]

    # 2) 컬럼명 변경 (행정구역(시군구)별 -> 시도, 2026.01 -> 인구)
    df = df.rename(columns={"행정구역(시군구)별": "시도", "2026.01": "인구"})

    # 3) 숫자 변환 (변환 불가능한 값은 NaN 처리)
    df["인구"] = pd.to_numeric(df["인구"], errors="coerce")

    out_path = f"{SOURCE_DIR}/population_sido.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("저장 완료:", out_path)
    print(df.head(20))
    return df


# =====================================================
# 5단계: 인구 대비 매장 수 (10만명당 매장 수) + 밀집 클러스터링
# =====================================================
def step5_density_analysis(store_count, df_pop):
    """
    3단계 매장수 + 4단계 인구(df_pop, 메모리에서 바로 전달받음)를 병합해
    '10만명당 매장수' 계산 후 source/hollys_population_analysis.csv 로 저장.
    (population_sido.csv를 다시 읽지 않음 -> 같은 실행 안에서는 방금 만든 df_pop 재사용)
    """
    df_merge = store_count.merge(df_pop, on="시도", how="inner")
    df_merge["10만명당_매장수"] = (df_merge["매장수"] / df_merge["인구"]) * 100000
    df_merge = df_merge.sort_values("10만명당_매장수", ascending=False)
    print(df_merge)

    out_path = f"{SOURCE_DIR}/hollys_population_analysis.csv"
    df_merge.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("저장 완료:", out_path)
    return df_merge


def step5b_cluster_dbscan():
    """
    위도/경도 기준 DBSCAN으로 0.8km 이내 매장 밀집 클러스터 탐지.
    source/hollys_cluster.csv 로 저장.
    """
    df = pd.read_csv(f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv")
    df = df.dropna(subset=["위도", "경도"]).reset_index(drop=True)

    coords = df[["위도", "경도"]].values
    kms_per_radian = 6371.0088
    epsilon = 0.8 / kms_per_radian  # 0.8km 이내 매장 밀집 기준

    db = DBSCAN(eps=epsilon, min_samples=5, algorithm="ball_tree", metric="haversine")
    df["cluster"] = db.fit_predict(np.radians(coords))

    print(df["cluster"].value_counts())
    out_path = f"{SOURCE_DIR}/hollys_cluster.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("저장 완료:", out_path)
    return df


# =====================================================
# 6단계: 기사용 순위표
# =====================================================
def step6_make_report(df_merge):
    """인구(만명) 컬럼을 추가한 보고서용 요약 테이블을 output/hollys_report.csv로 저장"""
    df_merge = df_merge.copy()
    df_merge["인구(만명)"] = df_merge["인구"] / 10000

    df_report = df_merge[["시도", "매장수", "인구(만명)", "10만명당_매장수"]].round(2)
    print(df_report)

    out_path = f"{OUTPUT_DIR}/hollys_report.csv"
    df_report.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("저장 완료:", out_path)
    return df_report


# =====================================================
# 7단계: 그래프 시각화 (막대그래프 / 산점도)
# =====================================================
def step7_plot_bar(df_merge):
    """시도별 인구 10만명당 매장 수 막대그래프 -> output/hollys_barplot.png"""
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_merge, hue="시도", x="시도", y="10만명당_매장수")

    plt.xticks(rotation=45)
    plt.title("시도별 인구 10만명당 할리스 매장 수")
    plt.xlabel("시도")
    plt.ylabel("10만명당 매장 수")

    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height(),
            f"{p.get_height():.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/hollys_barplot.png"
    plt.savefig(out_path, dpi=200)
    plt.show()
    print("저장 완료:", out_path)


def step7b_plot_scatter(df_merge):
    """시도별 인구 vs 매장 수 산점도 (파일 저장 없이 화면 출력만)"""
    plt.figure(figsize=(8, 6))
    plt.scatter(df_merge["인구"], df_merge["매장수"])

    for _, row in df_merge.iterrows():
        plt.text(row["인구"], row["매장수"], row["시도"], fontsize=9)

    plt.title("시도별 인구와 할리스 매장 수 관계")
    plt.xlabel("인구")
    plt.ylabel("매장 수")
    plt.tight_layout()
    plt.show()


# =====================================================
# 8단계: 지도 시각화 (Choropleth)
# =====================================================
def step8_choropleth_density():
    """
    시도별 10만명당 매장수를 Choropleth로 시각화 + 전체 매장 위치에 마커 표시.
    output/hollys_density_map.html 로 저장.
    """
    df_analysis = pd.read_csv(f"{SOURCE_DIR}/hollys_population_analysis.csv")
    df_stores = pd.read_csv(f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv")

    with open(f"{SOURCE_DIR}/skorea-provinces-2018-geo.json", encoding="utf-8") as f:
        geo = json.load(f)

    # 대한민국 전체가 보이도록 중심 좌표/줌 레벨 설정
    m = folium.Map(location=[36.5, 127.8], zoom_start=7)

    folium.Choropleth(
        geo_data=geo,
        data=df_analysis,
        columns=["시도", "10만명당_매장수"],
        key_on="feature.properties.name",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="10만명당 할리스 매장 수",
    ).add_to(m)

    for _, row in df_stores.iterrows():
        if pd.notna(row["위도"]) and pd.notna(row["경도"]):
            folium.Marker(
                location=[row["위도"], row["경도"]],
                popup=row["매장명"],
                tooltip=row["매장명"],
                icon=folium.Icon(color="red", icon="coffee", prefix="fa"),
            ).add_to(m)

    out_path = f"{OUTPUT_DIR}/hollys_density_map.html"
    m.save(out_path)
    print("저장 완료:", out_path)


def step8b_choropleth_gwangju():
    """광주광역시 매장만 확대해서 개별 매장 마커 지도 생성 -> output/hollys_gwangju_map.html"""
    df = pd.read_csv(f"{SOURCE_DIR}/hollys_store_geo_kakao_final.csv")
    df_gwangju = df[df["시도"].str.contains("광주광역시")].dropna(subset=["위도", "경도"])

    m = folium.Map(location=[35.13, 126.83], zoom_start=12)

    for _, row in df_gwangju.iterrows():
        tooltip_text = f"매장명: {row['매장명']}<br>주소: {row['주소']}<br>전화번호: {row['전화번호']}"
        popup_text = (
            f"<b>매장명:</b> {row['매장명']}<br>"
            f"<b>주소:</b> {row['주소']}<br>"
            f"<b>전화번호:</b> {row['전화번호']}"
        )

        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=5,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=folium.Tooltip(tooltip_text),
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    out_path = f"{OUTPUT_DIR}/hollys_gwangju_map.html"
    m.save(out_path)
    print("저장 완료:", out_path)


# =====================================================
# 실행부
# =====================================================
STEP_CHOICES = ["crawl", "geocode", "density", "cluster", "plot", "map", "all"]


def run(step: str):
    setup_env()

    if step in ("crawl", "all"):
        step1_crawl_stores()

    if step in ("geocode", "all"):
        step2_geocode()

    df_merge = None
    if step in ("density", "cluster", "plot", "map", "all"):
        store_count = step3_count_by_sido()
        df_pop = step4_prepare_population()
        df_merge = step5_density_analysis(store_count, df_pop)
        step6_make_report(df_merge)

    if step in ("cluster", "all"):
        step5b_cluster_dbscan()

    if step in ("plot", "all") and df_merge is not None:
        step7_plot_bar(df_merge)
        step7b_plot_scatter(df_merge)

    if step in ("map", "all"):
        step8_choropleth_density()
        step8b_choropleth_gwangju()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="할리스 매장 분석 파이프라인")
    parser.add_argument(
        "--step",
        choices=STEP_CHOICES,
        default="all",
        help=(
            "crawl=1단계 크롤링 / geocode=2단계 지오코딩 / "
            "density=3~6단계 밀도분석+보고서 / cluster=5-2단계 DBSCAN 클러스터링 / "
            "plot=그래프 시각화 / map=8단계 지도 시각화 / all=전체 실행"
        ),
    )
    args = parser.parse_args()
    run(args.step)