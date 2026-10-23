"""Human-readable labels for NIK 2026 model predictors.

This is a *display-only* translation of feature names. It does not change
input data, fitted models, SHAP values, or the sign of contributions.
"""
from __future__ import annotations

import re

FEATURE_LABELS = {
    "share_anthro": "share of human-modified ground",
    "share_bedrock": "share of exposed bedrock",
    "share_clay": "share of clay deposits",
    "share_fluvial_flood": "share of river and flood deposits",
    "share_glaciofluvial": "share of deposits carried by glacial meltwater",
    "share_marine": "share of marine deposits",
    "share_mass_movement": "share of mass-movement deposits",
    "share_moraine": "share of glacial till (moraine)",
    "share_organic": "share of organic deposits",
    "share_other": "share of other mapped deposits",
    "share_unknown": "share of unmapped or unclassified deposits",
    "share_water": "share of water-covered area",
    "share_weathering": "share of weathered material",
    "elev_mean": "average elevation",
    "elev_stdev": "variation in elevation",
    "elev_min": "minimum elevation",
    "elev_max": "maximum elevation",
    "relief": "terrain relief (elevation range)",
    "slp_mean": "average terrain slope",
    "slp_stdev": "variation in terrain slope",
    "slp_max": "maximum terrain slope",
    "slp_min": "minimum terrain slope",
    "slp_range": "range of terrain slopes",
    "river_length_km": "total mapped river length",
    "river_density": "density of mapped rivers",
    "tg_mean_2014_2023": "average air temperature (2014–2023)",
    "tg_change_2014_2023": "change in air temperature (2014–2023)",
    "rr_mean_2014_2023": "average precipitation (2014–2023)",
    "rr_change_2014_2023": "change in precipitation (2014–2023)",
}

# Replace entire feature identifiers only, keeping signs and numerical SHAP values.
_FEATURE_CODE = re.compile(r"\b(?:share|elev|slp|tg|rr|river)_[a-zA-Z0-9_]+\b")


def humanize_feature_text(value: object) -> str:
    """Replace feature codes anywhere in a SHAP factor string or saved summary."""
    if value is None:
        return ""
    text = str(value)
    return _FEATURE_CODE.sub(
        lambda m: FEATURE_LABELS.get(m.group(0), m.group(0).replace("_", " ")),
        text,
    )
