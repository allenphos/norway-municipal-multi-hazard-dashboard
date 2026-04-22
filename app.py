import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Norway Multi-Hazard Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# =========================
# CONFIG
# =========================
DATA_FILE = "df_mhi_with_llm.parquet"
GPKG_FILE = "municipalities_master.gpkg"
GPKG_LAYER = None
CREDS_FILE = "creds.json"

MODEL_NAME = "gpt-5.4-mini"

MAP_MODE_OPTIONS = {
    "Landslide probability": "landslide_prob_noflood",
    "Flood exposure": "flood_permille_clean",
    "Multi-hazard index": "mhi"
}

# =========================
# OPENAI CLIENT
# =========================


def load_openai_client():
    if not Path(CREDS_FILE).exists():
        return None

    with open(CREDS_FILE, "r", encoding="utf-8") as f:
        creds = json.load(f)

    api_key = creds.get("OPENAI_API_KEY")
    if not api_key:
        return None

    return OpenAI(api_key=api_key)


client = load_openai_client()

# =========================
# HELPERS
# =========================


@st.cache_data
def load_tabular_data(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError("Use parquet or csv.")


@st.cache_data
def load_geometries(path: str, layer=None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    gdf["kommunenummer"] = gdf["kommunenummer"].astype(
        str).str.strip().str.zfill(4)
    return gdf


def prob_band(p: float) -> str:
    if pd.isna(p):
        return "Missing"
    if p >= 0.90:
        return "High"
    if p >= 0.70:
        return "Relatively high"
    if p >= 0.50:
        return "Moderate"
    return "Lower"


def mhi_band(v: float) -> str:
    if pd.isna(v):
        return "Missing"
    if v >= 0.65:
        return "High"
    if v >= 0.50:
        return "Moderately high"
    if v >= 0.35:
        return "Moderate"
    if v >= 0.20:
        return "Moderately low"
    return "Lower"


def flood_band(v: float) -> str:
    if pd.isna(v):
        return "Missing"
    if v >= 0.50:
        return "Relatively elevated"
    if v >= 0.10:
        return "Moderate"
    return "Low"


def simplify_geom(gdf: gpd.GeoDataFrame, tolerance: float = 0.001) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(
        tolerance=tolerance, preserve_topology=True)
    return gdf


def fix_geometry_for_webmap(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def build_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """
    Build GeoJSON safely using only columns needed for mapping.
    Avoids Timestamp / mixed object serialization issues.
    """
    features = []

    for _, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": {
                "plot_id": str(row["plot_id"]),
                "kommunenavn": row.get("kommunenavn"),
                "kommunenummer": row.get("kommunenummer"),
                "landslide_prob_noflood": None if pd.isna(row.get("landslide_prob_noflood")) else float(row.get("landslide_prob_noflood")),
                "flood_permille_clean": None if pd.isna(row.get("flood_permille_clean")) else float(row.get("flood_permille_clean")),
                "mhi": None if pd.isna(row.get("mhi")) else float(row.get("mhi")),
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def get_bounds_center(gdf: gpd.GeoDataFrame):
    minx, miny, maxx, maxy = gdf.total_bounds
    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    lon_span = max(maxx - minx, 0.20)
    lat_span = max(maxy - miny, 0.20)

    return center_lon, center_lat, lon_span, lat_span


def build_single_prompt(row: pd.Series) -> str:
    prob = None if pd.isna(row["landslide_prob_noflood"]) else round(
        float(row["landslide_prob_noflood"]), 3)
    flood = None if pd.isna(row["flood_permille_clean"]) else round(
        float(row["flood_permille_clean"]), 3)
    mhi = None if pd.isna(row["mhi"]) else round(float(row["mhi"]), 3)

    top_pos = row.get("top_positive_shap", "Not available")
    top_neg = row.get("top_negative_shap", "Not available")

    landslide_level = prob_band(prob) if prob is not None else "Missing"
    flood_level = flood_band(flood) if flood is not None else "Missing"
    mhi_level = mhi_band(mhi) if mhi is not None else "Missing"

    return f"""
Municipality: {row['kommunenavn']}
Municipality number: {row['kommunenummer']}

Landslide probability value: {prob}
Landslide probability level: {landslide_level}

Flood exposure value (‰ of municipal area): {flood}
Flood exposure level: {flood_level}

Relative multi-hazard index value: {mhi}
Relative multi-hazard index level: {mhi_level}

Top positive local model contributors:
{top_pos}

Top negative local model contributors:
{top_neg}

Return valid JSON with exactly these keys:
- municipality
- summary

Rules for summary:
- Maximum 3 short sentences.
- Start with the municipality name.
- Use the qualitative levels above, not only the raw numbers.
- Prefer phrases like "high model-based landslide probability estimate", "low flood exposure", or "moderately high combined level".
- Explain the main model factors in simple language.
- Do not use the word SHAP.
- Do not add outside facts.
- Do not claim causality.
- Do not say that a landslide will happen.
- Keep the wording suitable for a public-facing dashboard.
""".strip()

def generate_llm_summary(row: pd.Series) -> str:
    if client is None:
        return "OpenAI client is not configured."

    prompt = build_single_prompt(row)

    instructions = """
    You are assisting a Norway municipal multi-hazard dashboard.
Use only the provided values and qualitative levels.
Prefer plain-language interpretation over repeating raw numbers.
Do not add outside facts.
Do not claim causality.
Do not mention future trends.
Do not use the word SHAP.
Output valid JSON only.
    """.strip()

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=prompt,
        max_output_tokens=220
    )

    raw = response.output_text.strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("summary", raw)
    except Exception:
        return raw


def landslide_plain_language(p: float) -> str:
    if pd.isna(p):
        return "No landslide estimate is available for this municipality."
    if p >= 0.95:
        return "This municipality is in the higher end of the landslide screening results."
    if p >= 0.90:
        return "This municipality shows a relatively elevated landslide screening result."
    if p >= 0.80:
        return "This municipality shows a moderate to relatively elevated landslide screening result."
    return "This municipality is in the lower part of the landslide screening results."


def flood_plain_language(v: float) -> str:
    if pd.isna(v):
        return "No flood exposure value is available."
    if v >= 0.12:
        return "Flood exposure is relatively elevated compared with many other municipalities."
    if v >= 0.06:
        return "Flood exposure is moderate."
    if v > 0:
        return "Flood exposure is present, but relatively low."
    return "No mapped overlap with flood caution areas is shown here."


# =========================
# LOAD DATA
# =========================
if not Path(DATA_FILE).exists():
    st.error(f"Data file not found: {DATA_FILE}")
    st.stop()

if not Path(GPKG_FILE).exists():
    st.error(f"GeoPackage file not found: {GPKG_FILE}")
    st.stop()

df = load_tabular_data(DATA_FILE)
gdf = load_geometries(GPKG_FILE, layer=GPKG_LAYER)

df["kommunenummer"] = df["kommunenummer"].astype(str).str.strip().str.zfill(4)

if "kommunenavn" in df.columns:
    df["kommunenavn"] = df["kommunenavn"].astype(str)

cols_to_merge = [
    "kommunenummer",
    "kommunenavn",
    "landslide_prob_noflood",
    "flood_permille_clean",
    "mhi",
    "top_positive_shap",
    "top_negative_shap",
    "llm_summary",
    "llm_caution_flag",
    "llm_parse_ok",
    "llm_model_used"
]

df_small = df[[c for c in cols_to_merge if c in df.columns]].copy()

gdf_map = gdf.to_crs(epsg=4326).copy()
#gdf_map = simplify_geom(gdf_map, tolerance=0.001)
gdf_map = gdf_map.merge(df_small, on="kommunenummer", how="left")

if "kommunenavn_x" in gdf_map.columns and "kommunenavn_y" in gdf_map.columns:
    gdf_map["kommunenavn"] = gdf_map["kommunenavn_y"].combine_first(
        gdf_map["kommunenavn_x"])
    gdf_map.drop(columns=["kommunenavn_x", "kommunenavn_y"], inplace=True)
elif "kommunenavn_x" in gdf_map.columns:
    gdf_map.rename(columns={"kommunenavn_x": "kommunenavn"}, inplace=True)
elif "kommunenavn_y" in gdf_map.columns:
    gdf_map.rename(columns={"kommunenavn_y": "kommunenavn"}, inplace=True)

gdf_map = gdf_map[gdf_map["kommunenavn"].notna()].copy()
gdf_map = gdf_map.reset_index(drop=True)
gdf_map["plot_id"] = gdf_map.index.astype(str)
gdf_map = fix_geometry_for_webmap(gdf_map)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Controls")

map_label = st.sidebar.selectbox("Map layer", list(MAP_MODE_OPTIONS.keys()))
map_col = MAP_MODE_OPTIONS[map_label]

municipality_list = sorted(gdf_map["kommunenavn"].dropna().unique().tolist())

if not municipality_list:
    st.error("No municipalities found after merge.")
    st.stop()

if "selected_muni" not in st.session_state:
    st.session_state["selected_muni"] = municipality_list[0]

if st.session_state["selected_muni"] not in municipality_list:
    st.session_state["selected_muni"] = municipality_list[0]

selected_from_sidebar = st.sidebar.selectbox(
    "Municipality",
    municipality_list,
    index=municipality_list.index(st.session_state["selected_muni"])
)
st.session_state["selected_muni"] = selected_from_sidebar

show_data_table = st.sidebar.checkbox("Show data table", value=False)
use_saved_llm = st.sidebar.checkbox(
    "Use saved LLM summary if available", value=True)
# zoom_to_selected = st.sidebar.checkbox("Zoom to selected municipality", value=True)

# =========================
# HEADER
# =========================
st.title("Norway Municipal Multi-Hazard Dashboard")
st.caption(
    "Municipality-level screening view combining model-estimated shallow landslide probability, "
    "flood exposure, and a relative multi-hazard index."
)

# =========================
# MAP DATA
# =========================
map_cols = [
    "plot_id",
    "kommunenummer",
    "kommunenavn",
    "landslide_prob_noflood",
    "flood_permille_clean",
    "mhi",
    "geometry"
]

plot_df = gdf_map[map_cols].copy()
geojson_data = build_geojson(plot_df)

# -------- Flood classes --------
flood_max = plot_df["flood_permille_clean"].max()
if pd.isna(flood_max):
    flood_max = 0.12

plot_df["flood_class"] = pd.cut(
    plot_df["flood_permille_clean"],
    bins=[-0.000001, 0, 0.03, 0.04, 0.06, 0.12, flood_max + 1e-9],
    labels=[
        "0 (no overlap)",
        "0–0.03 ‰",
        "0.03–0.04 ‰",
        "0.04–0.06 ‰",
        "0.06–0.12 ‰",
        "0.12+ ‰"
    ],
    include_lowest=True
)

flood_color_map = {
    "0 (no overlap)": "#f2f2f2",
    "0–0.03 ‰": "#dfe3f0",
    "0.03–0.04 ‰": "#c7cceb",
    "0.04–0.06 ‰": "#9ea5e5",
    "0.06–0.12 ‰": "#6b73d6",
    "0.12+ ‰": "#3f47b7"
}

# -------- Landslide classes --------
landslide_max = plot_df["landslide_prob_noflood"].max()
if pd.isna(landslide_max):
    landslide_max = 0.97

plot_df["landslide_class"] = pd.cut(
    plot_df["landslide_prob_noflood"],
    bins=[0.33, 0.80, 0.90, 0.95, 0.96, 0.97, landslide_max + 1e-9],
    labels=[
        "0.33–0.80",
        "0.80–0.90",
        "0.90–0.95",
        "0.95–0.96",
        "0.96–0.97",
        "0.97+"
    ],
    include_lowest=True
)

plot_df["landslide_class"] = plot_df["landslide_class"].astype("object")
plot_df.loc[plot_df["landslide_prob_noflood"].isna(), "landslide_class"] = "Missing / excluded"

landslide_color_map = {
    "0.33–0.80": "#f6ead7",
    "0.80–0.90": "#f6c98b",
    "0.90–0.95": "#fb8d52",
    "0.95–0.96": "#e5371f",
    "0.96–0.97": "#8e0000",
    "0.97+": "#5c0000",
    "Missing / excluded": "#c9c9c9"
}

# -------- MHI classes --------
mhi_max = plot_df["mhi"].max()
if pd.isna(mhi_max):
    mhi_max = 0.65

plot_df["mhi_class"] = pd.cut(
    plot_df["mhi"],
    bins=[0, 0.20, 0.35, 0.50, 0.65, mhi_max + 1e-9],
    labels=[
        "0.00–0.20",
        "0.20–0.35",
        "0.35–0.50",
        "0.50–0.65",
        "0.65+"
    ],
    include_lowest=True
)

plot_df["mhi_class"] = plot_df["mhi_class"].astype("object")
plot_df.loc[plot_df["mhi"].isna(), "mhi_class"] = "Missing"

mhi_color_map = {
    "0.00–0.20": "#f6ead7",
    "0.20–0.35": "#f3c98b",
    "0.35–0.50": "#f58d52",
    "0.50–0.65": "#e5371f",
    "0.65+": "#8e0000",
    "Missing": "#c9c9c9"
}

selected_outline = plot_df[plot_df["kommunenavn"] == st.session_state["selected_muni"]].copy()

# =========================
# BUILD MAP
# =========================
fig = go.Figure()

if map_col == "landslide_prob_noflood":
    class_order = [
        "0.33–0.80",
        "0.80–0.90",
        "0.90–0.95",
        "0.95–0.96",
        "0.96–0.97",
        "0.97+",
        "Missing / excluded"
    ]
    class_col = "landslide_class"
    color_map = landslide_color_map
    line_color = "#8c8c8c"

elif map_col == "flood_permille_clean":
    class_order = [
        "0 (no overlap)",
        "0–0.03 ‰",
        "0.03–0.04 ‰",
        "0.04–0.06 ‰",
        "0.06–0.12 ‰",
        "0.12+ ‰"
    ]
    class_col = "flood_class"
    color_map = flood_color_map
    line_color = "#8c8c8c"

else:
    class_order = [
        "0.00–0.20",
        "0.20–0.35",
        "0.35–0.50",
        "0.50–0.65",
        "0.65+",
        "Missing"
    ]
    class_col = "mhi_class"
    color_map = mhi_color_map
    line_color = "#8c8c8c"

for cls in class_order:
    class_df = plot_df[plot_df[class_col] == cls].copy()
    if class_df.empty:
        continue

    class_geojson = build_geojson(class_df)

    fig.add_trace(
        go.Choropleth(
            geojson=class_geojson,
            locations=class_df["plot_id"],
            z=[1] * len(class_df),
            featureidkey="properties.plot_id",
            customdata=class_df[
                [
                    "kommunenavn",
                    "kommunenummer",
                    "landslide_prob_noflood",
                    "flood_permille_clean",
                    "mhi"
                ]
            ],
            colorscale=[[0, color_map[cls]], [1, color_map[cls]]],
            showscale=False,
            name=cls,
            showlegend=False,
            marker_line_width=0.45,
            marker_line_color=line_color,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Municipality no.: %{customdata[1]}<br>"
                "Landslide probability: %{customdata[2]:.3f}<br>"
                "Flood exposure: %{customdata[3]:.3f}<br>"
                "Relative MHI: %{customdata[4]:.3f}<extra></extra>"
            )
        )
    )

# selected municipality outline
if not selected_outline.empty:
    selected_geojson = build_geojson(selected_outline)

    fig.add_trace(
        go.Choropleth(
            geojson=selected_geojson,
            locations=selected_outline["plot_id"],
            z=[1] * len(selected_outline),
            featureidkey="properties.plot_id",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,255,255,0.12)"]],
            showscale=False,
            marker_line_width=2.2,
            marker_line_color="cyan",
            hoverinfo="skip",
            name="Selected municipality",
            showlegend=False
        )
    )

fig.update_geos(
    fitbounds="locations",
    visible=False,
    projection_type="mercator",
    showcountries=False,
    showcoastlines=False,
    showland=True,
    landcolor="#e6e6e6",
    bgcolor="#e6e6e6"
)

fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=980,
    showlegend=False
)

# =========================
# LAYOUT
# =========================
left, right = st.columns([2.35, 1])

with left:
    st.subheader(map_label)

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        key="muni_map",
        on_select="rerun",
        selection_mode=("points",),
        config={
            "scrollZoom": True,
            "displayModeBar": False,
            "displaylogo": False
        }
    )

    if event and "selection" in event and event["selection"].get("points"):
        point = event["selection"]["points"][0]
        clicked_plot_id = str(point["location"])
        clicked_match = plot_df.loc[plot_df["plot_id"] == clicked_plot_id, "kommunenavn"]

        if not clicked_match.empty:
            st.session_state["selected_muni"] = clicked_match.iloc[0]
            st.rerun()

selected_rows = gdf_map[gdf_map["kommunenavn"] == st.session_state["selected_muni"]]

if selected_rows.empty:
    st.error("Selected municipality was not found.")
    st.stop()

selected_row = selected_rows.iloc[0]

with right:
    st.subheader(st.session_state["selected_muni"])

    # manual legend
    if map_col == "landslide_prob_noflood":
        st.markdown("**Relative probability class**")
        st.markdown("""
        <div style="line-height:1.9; font-size:15px;">
            <span style="display:inline-block;width:16px;height:16px;background:#f6ead7;border:1px solid #999;margin-right:8px;"></span>0.33–0.80<br>
            <span style="display:inline-block;width:16px;height:16px;background:#f6c98b;border:1px solid #999;margin-right:8px;"></span>0.80–0.90<br>
            <span style="display:inline-block;width:16px;height:16px;background:#fb8d52;border:1px solid #999;margin-right:8px;"></span>0.90–0.95<br>
            <span style="display:inline-block;width:16px;height:16px;background:#e5371f;border:1px solid #999;margin-right:8px;"></span>0.95–0.96<br>
            <span style="display:inline-block;width:16px;height:16px;background:#8e0000;border:1px solid #999;margin-right:8px;"></span>0.96–0.97<br>
            <span style="display:inline-block;width:16px;height:16px;background:#5c0000;border:1px solid #999;margin-right:8px;"></span>0.97+<br>
            <span style="display:inline-block;width:16px;height:16px;background:#c9c9c9;border:1px solid #999;margin-right:8px;"></span>Missing / excluded
        </div>
        """, unsafe_allow_html=True)

    elif map_col == "flood_permille_clean":
        st.markdown("**Flood exposure class**")
        st.markdown("""
        <div style="line-height:1.9; font-size:15px;">
            <span style="display:inline-block;width:16px;height:16px;background:#f2f2f2;border:1px solid #999;margin-right:8px;"></span>0 (no overlap)<br>
            <span style="display:inline-block;width:16px;height:16px;background:#dfe3f0;border:1px solid #999;margin-right:8px;"></span>0–0.03 ‰<br>
            <span style="display:inline-block;width:16px;height:16px;background:#c7cceb;border:1px solid #999;margin-right:8px;"></span>0.03–0.04 ‰<br>
            <span style="display:inline-block;width:16px;height:16px;background:#9ea5e5;border:1px solid #999;margin-right:8px;"></span>0.04–0.06 ‰<br>
            <span style="display:inline-block;width:16px;height:16px;background:#6b73d6;border:1px solid #999;margin-right:8px;"></span>0.06–0.12 ‰<br>
            <span style="display:inline-block;width:16px;height:16px;background:#3f47b7;border:1px solid #999;margin-right:8px;"></span>0.12+ ‰
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("**Relative MHI class**")
        st.markdown("""
        <div style="line-height:1.9; font-size:15px;">
            <span style="display:inline-block;width:16px;height:16px;background:#f6ead7;border:1px solid #999;margin-right:8px;"></span>0.00–0.20<br>
            <span style="display:inline-block;width:16px;height:16px;background:#f3c98b;border:1px solid #999;margin-right:8px;"></span>0.20–0.35<br>
            <span style="display:inline-block;width:16px;height:16px;background:#f58d52;border:1px solid #999;margin-right:8px;"></span>0.35–0.50<br>
            <span style="display:inline-block;width:16px;height:16px;background:#e5371f;border:1px solid #999;margin-right:8px;"></span>0.50–0.65<br>
            <span style="display:inline-block;width:16px;height:16px;background:#8e0000;border:1px solid #999;margin-right:8px;"></span>0.65+<br>
            <span style="display:inline-block;width:16px;height:16px;background:#c9c9c9;border:1px solid #999;margin-right:8px;"></span>Missing
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.info(landslide_plain_language(selected_row["landslide_prob_noflood"]))
    st.caption(flood_plain_language(selected_row["flood_permille_clean"]))

    c1, c2 = st.columns(2)
    with c1:
        ls_prob = selected_row.get("landslide_prob_noflood", None)
        if pd.notna(ls_prob):
            st.metric("Landslide probability", f"{ls_prob:.3f}")
            st.caption(f"{prob_band(ls_prob)} relative probability")
        else:
            st.metric("Landslide probability", "NA")
            st.caption("Missing")

    with c2:
        flood_val = selected_row.get("flood_permille_clean", None)
        if pd.notna(flood_val):
            st.metric("Flood exposure (‰)", f"{flood_val:.3f}")
            st.caption(f"{flood_band(flood_val)} flood exposure")
        else:
            st.metric("Flood exposure (‰)", "NA")
            st.caption("Missing")

    mhi_val = selected_row.get("mhi", None)
    if pd.notna(mhi_val):
        st.metric("Relative MHI", f"{mhi_val:.3f}")
        st.caption(f"{mhi_band(mhi_val)} combined level")
    else:
        st.metric("Relative MHI", "NA")
        st.caption("Missing")

    with st.expander("Why the landslide estimate looks like this", expanded=True):
        st.markdown("**What increased the landslide estimate**")
        st.write(selected_row.get("top_positive_shap", "Not available"))

        st.markdown("**What reduced the landslide estimate**")
        st.write(selected_row.get("top_negative_shap", "Not available"))

        st.caption(
            "This section explains which local factors had the strongest influence on the model result "
            "for this municipality. These are model signals, not proof that a landslide will happen."
        )

    st.subheader("Explanation")

    saved_summary = selected_row.get("llm_summary", None)

    if use_saved_llm and pd.notna(saved_summary) and str(saved_summary).strip():
        st.info(saved_summary)
    else:
        if st.button("Generate explanation"):
            with st.spinner("Generating explanation..."):
                try:
                    llm_text = generate_llm_summary(selected_row)
                    st.success("Done")
                    st.write(llm_text)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

if show_data_table:
    st.subheader("Underlying data")
    cols = [
        "kommunenummer",
        "kommunenavn",
        "landslide_prob_noflood",
        "flood_permille_clean",
        "mhi",
        "top_positive_shap",
        "top_negative_shap",
        "llm_summary"
    ]
    cols = [c for c in cols if c in gdf_map.columns]
    st.dataframe(gdf_map[cols], use_container_width=True)

st.subheader("Additional information")
st.write("For more detailed information, please see the official maps and source links below.")

st.markdown("""
- [NVE Hazard Map](https://temakart.nve.no/tema/faresoner)
- [seNorge](https://senorge.no/)
""")