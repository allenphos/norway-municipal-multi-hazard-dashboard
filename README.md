# Norway Municipal Multi-Hazard Dashboard

This repository contains my bachelor project on municipal-scale multi-hazard screening in Norway . The project combines a model-based shallow landslide probability estimate, municipal flood exposure, and a relative multi-hazard index in an interactive Streamlit dashboard.

Live app: https://norway-municipal-multi-hazard-dashboard-72rc6fzb6lrs9harg2vash.streamlit.app/

The goal is not to predict exact future events for a specific location and time. Instead, the project provides a municipality-level screening view that helps compare relative patterns across Norway.

Note: If the app is sleeping, please click the “Wake up” button and allow a short moment for it to load.

## Project overview

The project includes three main components:

- a machine learning model for shallow landslide probability at municipality level
- a flood exposure indicator based on overlap with mapped flood caution areas
- a combined relative multi-hazard index

The dashboard also includes plain-language explanations based on model outputs and local feature contributions.

## Main outputs

The repository contains:

- `app.py` — Streamlit dashboard
- `notebooks/bachelor_multihazard_workflow.ipynb` — main analysis notebook
- `data/df_mhi_with_llm.parquet` — processed municipality-level outputs
- `data/municipalities_master.gpkg` — municipality geometry

## Methods summary

The landslide component is based on a municipality-level classification framework using terrain, deposit, hydrological, and climate-related predictors. A calibrated tree-based model was used to estimate relative shallow landslide probability.

Flood was represented as a municipal exposure indicator rather than a probabilistic flood forecast.

The final multi-hazard index combines min-max normalised landslide probability and flood exposure into a relative screening measure.

Model interpretation was supported by local feature contribution analysis, and the dashboard includes plain-language summaries built from already computed outputs.

## Dashboard features

The Streamlit app allows the user to:

- view landslide probability, flood exposure, or relative multi-hazard index on a map
- select a municipality
- inspect model factors that pushed the landslide estimate upward or downward
- read a plain-language summary for the selected municipality

## Repository structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── interim/
│   │   ├── flood_feature.parquet
│   │   ├── hydrology.parquet
│   │   ├── model_dataset_loss_elev_slp_yi.parquet
│   │   ├── model_features_clean.parquet
│   │   ├── multihazard.parquet
│   │   ├── municipalities_climate.parquet
│   │   └── precipitations_municipalities.parquet
│   ├── df_mhi_with_llm.parquet
│   └── municipalities_master.gpkg
├── notebooks/
│   └── bachelor_multihazard_workflow.ipynb
```
The data/interim/ folder contains intermediate processed datasets used during feature construction and modelling workflow development.


## How to run locally

Create and activate a virtual environment if needed, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:
```bash
streamlit run app.py
```

## Data notes

This repository uses processed project outputs for the dashboard. Some large raw source datasets may be excluded from the repository because of file size.

Core source families used in the project include:

- Kartverket municipality boundaries and terrain data
- NVE landslide and flood-related data
- NGU surface deposit data
- seNorge climate data
- Important interpretation note

The dashboard is a screening tool. It does not provide event prediction, return periods, or official risk decisions. The results should be interpreted as relative model-based patterns at municipality level.


## Author

Anastasia Alyoshkina

Bachelor project in Applied Data Science

Noroff University College
