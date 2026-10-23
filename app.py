import html
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nik_feature_labels import humanize_feature_text

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
# Files exported by the final NIK 2026 notebook (Sections 5, 7, and 9).
# Do not use the old df_mhi_with_llm.parquet for hazard values.
DATA_FILE = "results/mhi_pipeline_final_355.csv"
FLOOD_FILE = "results/flood_exposure_corrected_2026.csv"
GPKG_FILE = "data/interim/municipalities_land_pipeline_final.gpkg"

# Optional explanations, only if recomputed using the final analysis.
# Never silently merge the legacy df_mhi_with_llm.parquet summaries.
EXPLANATIONS_FILE = "results/municipality_explanations_final.csv"
SUMMARIES_FILE = "results/municipality_llm_summaries_final.csv"
GPKG_LAYER = None
MAP_MODE_OPTIONS = {
    "Landslide probability": "landslide_prob_noflood",
    "Flood exposure": "flood_permille_final",
    "Multi-hazard index": "mhi"
}

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
    if v > 300:
        return ">300‰"
    if v > 200:
        return "200–300‰"
    if v > 100:
        return "100–200‰"
    if v > 50:
        return "50–100‰"
    return "0–50‰"


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
                "flood_permille_final": None if pd.isna(row.get("flood_permille_final")) else float(row.get("flood_permille_final")),
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
        return "No corrected flood exposure value is available."
    if v == 0:
        return "No mapped overlap with flood caution areas in this dataset."
    return (f"Mapped flood-zone exposure: {v:.3f}‰ of municipal area "
            f"({flood_band(v)} class). This is not a flood probability.")


# =========================
# LOAD DATA: ONE CANONICAL FINAL SOURCE
# =========================
required_paths = [DATA_FILE, FLOOD_FILE, GPKG_FILE]
missing_paths = [path for path in required_paths if not Path(path).exists()]
if missing_paths:
    st.error("Required final NIK 2026 exports are missing: " + ", ".join(missing_paths))
    st.info(
        "Run Sections 5, 7 and 9 of NIK_2026_final_GitHub_publication_style.ipynb "
        "and copy the exported CSV/GeoPackage files into these paths. "
        "Old flood_permille_clean data cannot be substituted."
    )
    st.stop()


def normalize_ids(frame):
    frame = frame.copy()
    frame["kommunenummer"] = (
        frame["kommunenummer"].astype("string").str.strip()
        .str.replace(r"\.0$", "", regex=True).str.zfill(4)
    )
    if frame["kommunenummer"].isna().any() or frame["kommunenummer"].duplicated().any():
        raise ValueError("Null or duplicated municipality IDs in a final input")
    return frame


@st.cache_data(show_spinner=False)
def load_final_data(explanations_version, summaries_version):
    model_data = normalize_ids(load_tabular_data(DATA_FILE))
    flood_data = normalize_ids(load_tabular_data(FLOOD_FILE))
    geometry = normalize_ids(load_geometries(GPKG_FILE, layer=GPKG_LAYER))

    if len(model_data) != 355 or len(flood_data) != 357 or len(geometry) != 357:
        raise ValueError(
            "Unexpected municipality counts: expected 355 modelled, "
            "357 flood and 357 geometries; got "
            f"{len(model_data)}, {len(flood_data)}, {len(geometry)}"
        )

    for field in ("landslide_prob_noflood", "mhi"):
        if field not in model_data.columns:
            raise ValueError(f"Missing {field} in {DATA_FILE}")
    if "flood_permille_final" not in flood_data.columns:
        raise ValueError(f"Missing corrected flood_permille_final in {FLOOD_FILE}")
    if "kommunenavn" not in flood_data.columns:
        raise ValueError("Corrected flood CSV needs kommunenavn")
    if geometry.crs is None:
        raise ValueError("Map GeoPackage has no CRS")

    # Geometry exported by the notebook already has older value columns.
    # Drop those columns and always join freshly exported final numbers.
    geometry = geometry[["kommunenummer", "geometry"]].copy()
    flood_data["flood_permille_final"] = pd.to_numeric(
        flood_data["flood_permille_final"], errors="raise"
    )
    model_data["landslide_prob_noflood"] = pd.to_numeric(
        model_data["landslide_prob_noflood"], errors="raise"
    )
    model_data["mhi"] = pd.to_numeric(model_data["mhi"], errors="raise")

    if flood_data["flood_permille_final"].isna().any():
        raise ValueError("Missing corrected flood exposure values")
    if not model_data[["landslide_prob_noflood", "mhi"]].notna().all().all():
        raise ValueError("Missing landslide or MHI values in the final modelling output")
    if not model_data["landslide_prob_noflood"].between(0, 1).all():
        raise ValueError("Landslide probabilities outside [0, 1]")
    if not model_data["mhi"].between(0, 1).all():
        raise ValueError("Relative MHI values outside [0, 1]")
    if set(geometry["kommunenummer"]) != set(flood_data["kommunenummer"]):
        raise ValueError("Flood and map geometries have different municipality IDs")
    if set(model_data["kommunenummer"]) != set(flood_data["kommunenummer"]) - {"1856", "1144"}:
        raise ValueError("Modelled municipality IDs do not match the two documented exclusions")

    # The final MHI output also contains flood_permille_final. Confirm it
    # matches the separately exported 357-municipality flood audit exactly.
    if "flood_permille_final" in model_data.columns:
        comparison = model_data[["kommunenummer", "flood_permille_final"]].merge(
            flood_data[["kommunenummer", "flood_permille_final"]],
            on="kommunenummer", validate="one_to_one", suffixes=("_mhi", "_flood")
        )
        if not np.allclose(
            comparison["flood_permille_final_mhi"].astype(float),
            comparison["flood_permille_final_flood"].astype(float),
            rtol=0, atol=1e-6
        ):
            raise ValueError("MHI and corrected flood CSVs contain different flood values")

    if not np.isclose(flood_data["flood_permille_final"].max(), 431.894, atol=0.015):
        raise ValueError(
            "The corrected flood data do not match the NIK 2026 maximum "
            "of 431.894‰; verify the exported file before publishing."
        )

    merged = geometry.merge(
        flood_data[["kommunenummer", "kommunenavn", "flood_permille_final"]],
        on="kommunenummer", how="left", validate="one_to_one"
    ).merge(
        model_data[["kommunenummer", "landslide_prob_noflood", "mhi"]],
        on="kommunenummer", how="left", validate="one_to_one"
    )
    if len(merged) != 357:
        raise ValueError("Geometry merge changed municipality count")
    if (merged["flood_permille_final"].notna().sum() != 357
            or merged["landslide_prob_noflood"].notna().sum() != 355
            or merged["mhi"].notna().sum() != 355):
        raise ValueError("Final hazard values could not be matched by municipality ID")

    # Optional *updated* local explanations. Do not import outdated summaries
    # from the old data/df_mhi_with_llm.parquet file.
    if Path(EXPLANATIONS_FILE).exists():
        explanations = normalize_ids(load_tabular_data(EXPLANATIONS_FILE))
        extra_cols = [c for c in (
            "top_positive_shap", "top_negative_shap", "llm_summary",
            "llm_caution_flag", "llm_parse_ok", "llm_model_used"
        ) if c in explanations.columns]
        if extra_cols:
            if explanations["kommunenummer"].duplicated().any():
                raise ValueError("Duplicate municipality IDs in local SHAP file")
            if not set(explanations["kommunenummer"]) == set(model_data["kommunenummer"]):
                raise ValueError("Local SHAP IDs do not match the final 355 municipalities")
            merged = merged.merge(
                explanations[["kommunenummer", *extra_cols]],
                on="kommunenummer", how="left", validate="one_to_one"
            )

    # Summaries were pre-generated once via generate_nik_summaries.py.
    # The public dashboard NEVER calls a paid API.
    if Path(SUMMARIES_FILE).exists():
        summaries = normalize_ids(pd.read_csv(
            SUMMARIES_FILE, dtype={"kommunenummer": "string"}, keep_default_na=False
        ))
        expected_cols = {
            "kommunenummer", "llm_summary",
            "landslide_prob_noflood_source", "flood_permille_final_source", "mhi_source",
            "top_positive_shap_source", "top_negative_shap_source",
        }
        if expected_cols - set(summaries.columns):
            raise ValueError("Cached summaries CSV has missing provenance columns: "
                             + ", ".join(sorted(expected_cols - set(summaries.columns))))
        if not set(summaries["kommunenummer"]).issubset(set(model_data["kommunenummer"])):
            raise ValueError("Cached summary CSV includes unexpected municipality IDs")
        if summaries["llm_summary"].eq("").any():
            raise ValueError("Cached summary CSV contains blank summaries")
        # Ensure cached text belongs to exactly these numerical inputs.
        reference = merged.set_index("kommunenummer")
        check = summaries.set_index("kommunenummer")
        for source_col, final_col in (
            ("landslide_prob_noflood_source", "landslide_prob_noflood"),
            ("flood_permille_final_source", "flood_permille_final"),
            ("mhi_source", "mhi"),
        ):
            actual = pd.to_numeric(check[source_col], errors="raise").to_numpy()
            expected = reference.loc[check.index, final_col].astype(float).to_numpy()
            if not np.allclose(actual, expected, rtol=0, atol=1e-9):
                raise ValueError(
                    f"Stale cached summaries: {source_col} differs from the final hazard CSV. "
                    "Rerun generate_nik_summaries.py before publishing."
                )
        # SHAP provenance matters: a changed explanation must not silently
        # retain the wording generated for a previous fitted model.
        for source_col, final_col in (
            ("top_positive_shap_source", "top_positive_shap"),
            ("top_negative_shap_source", "top_negative_shap"),
        ):
            if final_col not in reference:
                raise ValueError(
                    "Cached AI explanations need results/municipality_explanations_final.csv "
                    "for model provenance verification."
                )
            actual = check[source_col].fillna("").astype(str).tolist()
            expected = reference.loc[check.index, final_col].fillna("").astype(str).tolist()
            if actual != expected:
                raise ValueError(
                    "Cached summaries do not match the final SHAP features. "
                    "Regenerate changed rows before publishing."
                )
        merged = merged.merge(
            summaries[["kommunenummer", "llm_summary"]],
            on="kommunenummer", how="left", validate="one_to_one"
        )

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=geometry.crs)


try:
    # Cache refreshes when either optional explanation CSV is created/updated.
    explanation_version = (Path(EXPLANATIONS_FILE).stat().st_mtime_ns
                           if Path(EXPLANATIONS_FILE).exists() else None)
    summaries_version = (Path(SUMMARIES_FILE).stat().st_mtime_ns
                         if Path(SUMMARIES_FILE).exists() else None)
    gdf_map = load_final_data(explanation_version, summaries_version)
    # Simplify in projected metres, preserving topology (only map geometry,
    # never model features or municipality identifiers).
    if gdf_map.crs.is_projected:
        gdf_map["geometry"] = gdf_map.geometry.simplify(
            tolerance=70, preserve_topology=True
        )
    gdf_map = gdf_map.to_crs(epsg=4326)
    gdf_map = fix_geometry_for_webmap(gdf_map)
except Exception as exc:
    st.error(f"Cannot load final map data: {exc}")
    st.stop()

if len(gdf_map) != 357:
    st.error("One or more municipal geometries are missing after validation")
    st.stop()

gdf_map = gdf_map.reset_index(drop=True)
gdf_map["plot_id"] = gdf_map["kommunenummer"]  # stable across reruns

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

# =========================
# HEADER
# =========================
st.title("Norway Municipal Multi-Hazard Dashboard")
st.caption(
    "Municipality-level screening view combining model-estimated shallow landslide probability, "
    "flood exposure, and a relative multi-hazard index."
)

# =========================
# MAP DATA: PUBLICATION-STYLE COLOURS AND CLASSES
# =========================
plot_df = gdf_map[[
    "plot_id", "kommunenummer", "kommunenavn",
    "landslide_prob_noflood", "flood_permille_final", "mhi", "geometry"
]].copy()

# Flood: fixed exposure bins, exactly as in the final paper-style figure.
FLOOD_LABELS = ["0–50‰", "50–100‰", "100–200‰", "200–300‰", ">300‰"]
FLOOD_COLOURS = ["#eff5fa", "#c3d9ee", "#6baed6", "#2171b5", "#08306b"]

# Landslide and MHI: quintiles recomputed from the final observed values.
# These colour scales follow the supplied NIK 2026 map examples.
LANDSLIDE_COLOURS = ["#eeecf5", "#bfbedd", "#958fc3", "#6e5aa8", "#4b198b"]
MHI_COLOURS = ["#fff7ec", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"]


def quintile_classification(values, palette, indicator):
    observed = values.dropna().astype(float)
    edges = np.unique(np.quantile(observed, np.linspace(0, 1, 6)))
    if len(edges) < 3:
        raise ValueError(f"{indicator}: insufficient distinct values for a quantile map")
    class_count = len(edges) - 1
    codes = pd.cut(values, bins=edges, include_lowest=True, labels=False)
    # If quintile edges coincide, do not arbitrarily split ties.
    colour_indices = np.linspace(0, len(palette) - 1, class_count).round().astype(int)
    colours = [palette[i] for i in colour_indices]
    return codes, edges, colours


plot_df["ls_class"], ls_edges, ls_colours = quintile_classification(
    plot_df["landslide_prob_noflood"], LANDSLIDE_COLOURS, "Landslide probability"
)
plot_df["mhi_class"], mhi_edges, mhi_colours = quintile_classification(
    plot_df["mhi"], MHI_COLOURS, "MHI"
)
plot_df["flood_class"] = pd.cut(
    plot_df["flood_permille_final"],
    bins=[-np.inf, 50, 100, 200, 300, np.inf],
    labels=False, include_lowest=True
)

if map_col == "landslide_prob_noflood":
    class_col = "ls_class"
    legend = [
        (f"Q{i + 1}: {ls_edges[i]:.4f}–{ls_edges[i + 1]:.4f}", colour)
        for i, colour in enumerate(ls_colours)
    ]
    map_note = "Modelled municipality-level relative screening; not annual landslide event probability."
elif map_col == "flood_permille_final":
    class_col = "flood_class"
    legend = list(zip(FLOOD_LABELS, FLOOD_COLOURS))
    map_note = "NVE flood caution-zone area as ‰ of municipal area; not flood probability."
else:
    class_col = "mhi_class"
    if len(mhi_colours) == 5:
        mhi_labels = ["Lowest 20%", "20–40%", "40–60%", "60–80%", "Highest 20%"]
    else:
        mhi_labels = [f"Q{i + 1}" for i in range(len(mhi_colours))]
    legend = list(zip(mhi_labels, mhi_colours))
    map_note = "Equal-weight, min–max-normalised MHI; a relative index, not joint hazard probability."

# White gaps are intentional: two excluded municipalities have no modelled
# landslide/MHI values. Do not draw a grey base layer or a missing-value class.
selected_outline = plot_df.loc[
    (plot_df["kommunenavn"] == st.session_state["selected_muni"])
    & plot_df[map_col].notna()
].copy()

# =========================
# INTERACTIVE CHOROPLETH
# =========================
fig = go.Figure()

for class_number, (legend_label, colour) in enumerate(legend):
    class_df = plot_df.loc[plot_df[class_col] == class_number].copy()
    if class_df.empty:
        continue
    fig.add_trace(go.Choropleth(
        geojson=build_geojson(class_df),
        locations=class_df["plot_id"],
        z=np.ones(len(class_df)),
        featureidkey="properties.plot_id",
        customdata=class_df[[
            "kommunenavn", "kommunenummer", "landslide_prob_noflood",
            "flood_permille_final", "mhi"
        ]].to_numpy(),
        colorscale=[[0, colour], [1, colour]],
        showscale=False, showlegend=False,
        name=legend_label,
        marker_line_width=0.20,
        marker_line_color="#ffffff",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Municipality no.: %{customdata[1]}<br>"
            "Landslide probability: %{customdata[2]:.3f}<br>"
            "Flood exposure: %{customdata[3]:.3f}‰<br>"
            "Relative MHI: %{customdata[4]:.3f}<extra></extra>"
        )
    ))

if not selected_outline.empty:
    fig.add_trace(go.Choropleth(
        geojson=build_geojson(selected_outline),
        locations=selected_outline["plot_id"],
        z=np.ones(len(selected_outline)),
        featureidkey="properties.plot_id",
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        showscale=False, showlegend=False,
        marker_line_width=1.7,
        marker_line_color="#162b45",
        hoverinfo="skip",
        name="Selected municipality"
    ))

fig.update_geos(
    fitbounds="locations", visible=False,
    projection_type="mercator",
    showcountries=False, showcoastlines=False, showland=False,
    bgcolor="#ffffff"
)
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    height=850,
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    showlegend=False,
    dragmode="pan"
)

# =========================
# LAYOUT
# =========================
left, right = st.columns([2.35, 1])

with left:
    st.subheader(map_label)
    st.caption(map_note)

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

    # Legend uses the exact classes/colours that appear on the map.
    st.markdown("**Map legend**")
    legend_html = "<div style='line-height:1.9;font-size:14px'>" + "".join(
        "<div style='display:flex;align-items:center;gap:9px'>"
        f"<span style='display:inline-block;flex:0 0 16px;height:14px;"
        f"background:{colour};border:1px solid #d4d4d4'></span>"
        f"<span>{html.escape(label)}</span></div>"
        for label, colour in legend
    ) + "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)
    if map_col != "flood_permille_final":
        st.caption("Unfilled areas: municipalities excluded from modelling (no estimate).")

    st.divider()

    st.info(landslide_plain_language(selected_row["landslide_prob_noflood"]))
    st.caption(flood_plain_language(selected_row["flood_permille_final"]))

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
        flood_val = selected_row.get("flood_permille_final", None)
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

    with st.expander("Feature contributions (uncalibrated HGB)", expanded=False):
        # Explanations are obtained from the separately fitted uncalibrated
        # model in the notebook, not from the calibrated probabilities.
        positive = selected_row.get("top_positive_shap")
        negative = selected_row.get("top_negative_shap")
        if pd.notna(positive) and pd.notna(negative):
            st.markdown("**Factors increasing the uncalibrated model score**")
            st.write(humanize_feature_text(positive))
            st.markdown("**Factors decreasing the uncalibrated model score**")
            st.write(humanize_feature_text(negative))
            st.caption(
                "SHAP explains the separately fitted uncalibrated HGB's model output, "
                "not the displayed calibrated landslide probability. "
                "Feature contributions are associations, not causal effects."
            )
        else:
            st.info(
                "Local SHAP explanations are not available yet. Run the "
                "municipality-level SHAP export cell in the NIK notebook and "
                "copy results/municipality_explanations_final.csv into this project."
            )
        
        
    st.subheader("Explanation")

    saved_summary = selected_row.get("llm_summary", None)

    if pd.notna(saved_summary) and str(saved_summary).strip():
        st.info(humanize_feature_text(saved_summary))
        st.caption(
            "Pre-generated explanation using final NIK 2026 values. "
            "Feature contributions refer to a separate uncalibrated HGB, "
            "not to the calibrated probability or a causal effect."
        )
    elif pd.notna(selected_row.get("landslide_prob_noflood")):
        st.info(
            "No cached explanation yet. Generate it once with "
            "`python generate_nik_summaries.py --all` and add "
            "`results/municipality_llm_summaries_final.csv` to the project. "
            "The public dashboard does not call the API."
        )
    else:
        st.caption("No modelled landslide value for this excluded municipality.")

if show_data_table:
    st.subheader("Underlying data")
    cols = [
        "kommunenummer",
        "kommunenavn",
        "landslide_prob_noflood",
        "flood_permille_final",
        "mhi",
        "top_positive_shap",
        "top_negative_shap",
        "llm_summary"
    ]
    cols = [c for c in cols if c in gdf_map.columns]
    view = gdf_map[cols].copy()
    for label_col in ("top_positive_shap", "top_negative_shap", "llm_summary"):
        if label_col in view.columns:
            view[label_col] = view[label_col].map(
                lambda value: humanize_feature_text(value) if pd.notna(value) else value
            )
    st.dataframe(view, use_container_width=True)

st.subheader("Additional information")
st.write("For more detailed information, please see the official maps and source links below.")

st.markdown("""
- [NVE Hazard Map](https://temakart.nve.no/tema/faresoner)
- [seNorge](https://senorge.no/)
""")