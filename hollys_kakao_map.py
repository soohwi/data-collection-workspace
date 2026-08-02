"""
할리스 매장 밀도 지도 시각화
- 시도별 인구 대비 매장 밀도 (Choropleth 음영)
- 전국 매장 위치 (MarkerCluster 마커)
"""

import json
import os

import pandas as pd
import folium
from folium.plugins import MarkerCluster


# ── 데이터 로드 ──────────────────────────────────────────
# 시도별 분석 결과 (Choropleth 음영용 - 인구 10만명당 매장 수)
df_analysis = pd.read_csv("source/hollys_population_analysis.csv")

# 매장 단위 원본 데이터 (위도/경도 포함, 마커 찍기용)
df_stores = pd.read_csv("source/hollys_store_geo_kakao_final.csv")
df_stores = df_stores.dropna(subset=["위도", "경도"])  # 좌표 없는 매장 제외

# 시도 경계 geojson (Choropleth 배경 지도용)
with open("source/skorea-provinces-2018-geo.json", encoding="utf-8") as f:
    geo = json.load(f)


# ── 지도 생성 ────────────────────────────────────────────
# 대한민국 전체가 보이도록 중심 좌표/줌 레벨 조정 (광주 기준 X, 전국 기준 O)
m = folium.Map(location=[36.5, 127.8], zoom_start=7)

# 1) 시도별 음영 (Choropleth) - 인구 대비 매장 밀도
folium.Choropleth(
    geo_data=geo,
    data=df_analysis,
    columns=["시도", "10만명당_매장수"],
    key_on="feature.properties.name",
    fill_opacity=0.7,
    line_opacity=0.3,
    legend_name="10만명당 할리스 매장 수",
    name="시도별 밀도"
).add_to(m)

# 2) 전국 매장 위치 마커 (많은 매장을 클러스터로 묶어 성능/가독성 확보)
marker_cluster = MarkerCluster(name="전체 매장").add_to(m)

for _, row in df_stores.iterrows():
    tooltip_text = f"""
    매장명: {row['매장명']}<br>
    주소: {row['주소']}<br>
    전화번호: {row['전화번호']}
    """
    popup_text = f"""
    <b>매장명:</b> {row['매장명']}<br>
    <b>주소:</b> {row['주소']}<br>
    <b>전화번호:</b> {row['전화번호']}
    """

    folium.Marker(
        location=[row["위도"], row["경도"]],
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=folium.Tooltip(tooltip_text),
        icon=folium.Icon(color="red", icon="coffee", prefix="fa")
    ).add_to(marker_cluster)

# 음영/마커 레이어를 각각 껐다 켰다 할 수 있도록 컨트롤 추가
folium.LayerControl().add_to(m)


# ── 저장 ────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)
m.save("output/hollys_density_map.html")
print("저장 완료: output/hollys_density_map.html")