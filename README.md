# Norway Municipal Multi-Hazard Dashboard

An interactive, explainable multi-hazard screening framework for Norwegian municipalities, combining machine learning-based shallow landslide estimates, mapped flood exposure, and a relative multi-hazard index.

**[Explore the Interactive Dashboard](https://norway-municipal-multi-hazard-dashboard-72rc6fzb6lrs9harg2vash.streamlit.app/)**

*If the application is sleeping, click "Wake up" to start it.*

## About the Project

This project originated as a Bachelor’s thesis in Applied Data Science at Noroff University College and was subsequently revised for the NIK 2026 research paper *Explainable Municipal Multi-Hazard Screening for Shallow Landslides and Floods in Norway*.

The study investigates how environmental data and interpretable machine learning can support municipality-level hazard screening across Norway.

Rather than predicting when or where an individual disaster will occur, the framework identifies relative geographical patterns that may support preliminary screening and further investigation.

The final modelling dataset contains **355 Norwegian municipalities**.

## Methodology

### 1. Shallow Landslide Modelling

A calibrated HistGradientBoosting (HGB) classifier estimates municipality-level shallow landslide occurrence probability using environmental predictors, including:

- Terrain characteristics and elevation
- Surface deposits and geological conditions
- Hydrological features
- Climate and precipitation indicators

The model is evaluated using a held-out test set and spatial cross-validation to assess geographical generalisation.

**Held-out test performance**

| Model | ROC-AUC | Average Precision | Brier Score |
|---|---:|---:|---:|
| Dummy | 0.500 | 0.901 | 0.089 |
| Logistic Regression | 0.732 | 0.962 | 0.094 |
| Calibrated HGB | **0.815** | **0.974** | **0.075** |

Spatial cross-validation indicates substantial variation in model performance across geographical regions.

The estimates are intended for relative screening, not as annual landslide occurrence probabilities.

### 2. Flood Exposure Assessment

Flood exposure is calculated from the geographical overlap between municipality areas and mapped NVE flood caution zones.

The revised calculation addresses overlapping flood polygons to avoid double-counting the same land area.

Flood exposure is expressed in per mille (‰) of the municipal land and freshwater area.

The corrected dataset has a maximum observed exposure of **431.894‰**.

Unlike landslide probability, this indicator represents mapped spatial exposure, not the probability of a future flood event.

### 3. Multi-Hazard Index

The relative Multi-Hazard Index (MHI) combines:

- Calibrated shallow landslide probability
- Corrected flood-zone exposure

Both components are min–max normalised and combined with equal weights.

The resulting index supports relative comparisons between municipalities. It should not be interpreted as a joint disaster probability or comprehensive risk assessment.

### 4. Model Explainability

SHAP is used to investigate feature contributions to an independently trained, uncalibrated HGB model.

Global explanations illustrate which environmental features are most influential, while local explanations identify features associated with higher or lower model outputs for individual municipalities.

These explanations describe model behaviour rather than causal relationships.

Precomputed plain-language summaries make the results more accessible to non-technical users. The dashboard displays saved summaries without requiring a live LLM request for each visitor.

## Interactive Dashboard

The Streamlit application provides three interactive map layers:

| Map | Description |
|---|---|
| Landslide Probability | Calibrated municipality-level model estimates |
| Flood Exposure | Corrected mapped flood-zone exposure (‰) |
| Multi-Hazard Index | Combined relative screening index |

Users can select a municipality to explore its environmental screening results, view relevant model feature contributions, and read a plain-language interpretation.

Map visualisations use the final revised results. The two municipalities excluded from landslide modelling have no landslide or MHI estimates rather than imputed values.

## Repository Structure

The updated project uses the following principal files:

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── feature_labels.py
├── generate_nik_summaries.py
├── notebooks/
│   └── NIK_2026_final_GitHub.ipynb
├── data/
│   └── interim/
│       └── municipalities_land_pipeline_final.gpkg
└── results/
    ├── mhi_pipeline_final_355.csv
    ├── flood_exposure_corrected_2026.csv
    ├── municipality_explanations_final.csv
    └── municipality_llm_summaries_final.csv
```

The notebook also uses additional intermediate modelling datasets and source geospatial files. Some large source data may not be included in the public repository.

## Running the Dashboard Locally

Clone the repository and navigate to the project directory.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The dashboard reads precomputed results, so retraining the machine learning models or regenerating language-model summaries is not required for ordinary dashboard use.

## Data Sources

The research integrates environmental and administrative data from Norwegian public data providers, including:

- **NVE:** Landslide and flood-related geospatial data
- **NGU:** Surface deposit and geological data
- **Kartverket:** Municipality boundaries and geographical information
- **SSB:** Municipality area statistics
- **seNorge:** Climate and precipitation data

Data processing, feature construction, and modelling are documented in the research workflow.

## Limitations

This project is an exploratory research and screening tool.

The estimated landslide probabilities are not annual event probabilities. Mapped flood exposure does not represent a probabilistic flood forecast, and MHI is not an absolute risk score.

Spatial validation demonstrates limitations in geographical generalisation. Model feature contributions should not be interpreted as evidence of causality.

The dashboard does not replace official hazard assessments or site-specific geological investigations.

## Research

**Explainable Municipal Multi-Hazard Screening for Shallow Landslides and Floods in Norway**

Research manuscript prepared for NIK 2026.

## Author

**Anastasia Alyoshkina**

Bachelor of Applied Data Science  
Noroff University College, Norway