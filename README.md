# 💨 AirProject: Physics-Guided Spatio-Temporal Hybrid Framework for Air Quality Virtual Sensing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **"A Decoupled Spatio-Temporal Hybrid Framework with Physics-Guided Scaling for Air Quality Virtual Sensing in Indian Non-Attainment Airsheds"**

---

## 📌 Executive Summary

Urban air quality management across Indian non-attainment cities (under NCAP targets) faces two major challenges:
1. **The Spatial Coverage Deficit:** Ground monitoring networks (CPCB CAAQMS) are sparse and often concentrated in industrial or specific municipal zones, leaving residential and suburban zones without real-time coverage.
2. **The Peak-Smoothing Failure:** Standard deep learning models trained on symmetric loss functions (like standard MSE or MAE) underpredict extreme, episodic pollution surges (such as winter smog and biomass burning spikes) by collapsing predictions toward the historical mean.

**AirProject** introduces a lightweight, computationally efficient two-stage hybrid framework for hourly $\text{PM}_{2.5}$ estimation and diurnal peak prediction ($T+1$ to $T+24$). It incorporates physics-guided feature engineering ($1/\text{PBLH}$ mixing factors, continuous $U/V$ wind vectors) and optimizes using Frequency-Weighted Mean Absolute Error ($\mathcal{L}_{fMAE}$) to prevent peak attenuation.

---

## 🏗️ Architecture Overview

The framework fuses three open-source data streams: **CPCB CAAQMS ground monitors**, **Open-Meteo / ERA5 reanalysis weather**, and **Sentinel-5P satellite observations** via Google Earth Engine.

```
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│       CPCB Ground       │     │    Open-Meteo / ERA5    │     │   Sentinel-5P (GEE)     │
│   (PM10, PM2.5, NO2)    │     │ (Temp, RH, Press, PBLH) │     │ (Aerosol Index, NO2 VCD)│
└────────────┬────────────┘     └────────────┬────────────┘     └────────────┬────────────┘
             │                               │                               │
             └───────────────────────┬───────┴───────────────────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │    Physics Feature Engine    │
                      │  - U/V Wind Transformations  │
                      │  - 1/PBLH Mixing Factor      │
                      │  - Cyclic Time Encoding      │
                      └──────────────────────────────┘
                                     │
 ┌───────────────────────────────────┴───────────────────────────────────┐
 │                                                                       │
 ▼                                                                       ▼
┌─────────────────────────────────┐                     ┌───────────────────────────────────┐
│  STAGE 1: Daily Spatial Model   │                     │  STAGE 2: Diurnal Generator       │
│  (Spatial Baseline Regressor)   │ ── Daily Baseline ─►│  (1D-CNN + BiLSTM + Attention)    │ ──► T+1..T+24
│  Satellite + Daily Weather      │      Constraint     │  Hourly Weather + U/V + 1/PBLH    │     Forecasts
└─────────────────────────────────┘                     └───────────────────────────────────┘
```

### Key Highlights
- **Decoupled 2-Stage Design:** Resolves the temporal mismatch caused by Sentinel-5P's single daily overpass (~1:30 PM solar time) by estimating a robust daily background level in Stage 1 and predicting hourly diurnal ratios in Stage 2.
- **Physics-Guided Preprocessing:**
  - **Trigonometric Wind Vectors ($U/V$):** Eliminates $0^\circ/360^\circ$ angular discontinuities:
    $$U = \text{WS} \times \cos\left(\theta \frac{\pi}{180}\right), \quad V = \text{WS} \times \sin\left(\theta \frac{\pi}{180}\right)$$
  - **Planetary Boundary Layer Height ($1/\text{PBLH}$):** Incorporates vertical dispersion physics to link satellite column densities to ground-level breathing zones.
- **Peak-Aware Optimization ($\mathcal{L}_{fMAE}$):** Dynamically weights samples inversely proportional to concentration bin frequencies so that rare, hazardous winter peaks receive higher gradient priority instead of being smoothed out.
- **Zero-Shot Spatial Generalization:** Evaluated using Leave-One-Station-Out (LOSO) and cross-city holdout protocols (e.g. testing directly on unseen cities like Lucknow and Patna) to prove true virtual sensing capability without spatial data leakage.

---

## 📊 Key Results

| Model / Configuration | Evaluation Protocol | $R^2$ Score | Key Benefit |
|-----------------------|---------------------|-------------|-------------|
| Stage 1 Baseline Only | Zero-Shot Cross-City | 0.4890 | Spatial background estimation |
| **Full Hybrid (Stage 1 + Stage 2)** | **Zero-Shot Cross-City** | **0.6119** | **+25.1% relative improvement** |
| Peak Event Reconstruction | Extreme Winter Smog | **846.2 $\mu g/m^3$ pred** vs. **830.0 $\mu g/m^3$ obs** | Captures 6.725× diurnal spike |

---

## 📂 Repository Structure

```
airproject/
├── configs/                   # Pipeline and station YAML configurations
│   ├── pipeline.yaml          # Hyperparameters and feature settings
│   └── stations.yaml          # Station coordinates and metadata
├── data/
│   ├── raw/                   # Raw CPCB and meteorology CSVs
│   ├── processed/             # Cleaned and feature-engineered datasets
│   └── final_multimodal/      # Fused multimodal station-level datasets
├── models/
│   └── saved/                 # Trained model checkpoints & scalers (.pth, .joblib)
├── paper/                     # Publication LaTeX manuscript & figures
│   ├── sections/              # Manuscript text sections
│   ├── figures/               # Generated publication-quality figures
│   └── paper_ready_for_overleaf.zip # Complete Overleaf-ready bundle
├── results/                   # Diagnostic plots and operational outputs
├── src/                       # Core codebase
│   ├── clean_cpcb.py          # Ground data cleaning & quality control
│   ├── fetch_cpcb.py          # OpenAQ / CPCB data ingestion
│   ├── fetch_openmeteo.py     # Weather & PBLH feature fetcher
│   ├── generate_stage1_daily.py # Stage 1 daily dataset generator
│   └── models/
│       ├── architecture.py    # Hybrid 1D-CNN + BiLSTM + Attention model
│       ├── dataloaders.py     # Windowed sequence loaders
│       ├── dataset.py         # PyTorch dataset implementations
│       ├── evaluate.py        # Model evaluation routines
│       ├── losses.py          # Peak-aware loss functions (LfMAE)
│       └── train.py           # Training loop & checkpointing
├── tests/                     # Automated unit and integration tests
├── dashboard.py               # Interactive Streamlit dashboard
├── eval_in_domain.py          # In-domain evaluation script
├── check_presets.py           # Preset validation script
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/pavan67-git/airproject.git
cd airproject

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch streamlit matplotlib scikit-learn
```

### 2. Launch the Interactive Dashboard

An interactive dashboard is included for exploring model predictions, diurnal profiles, peak analysis, and city-wide performance:

```bash
streamlit run dashboard.py
```

### 3. Run Pipeline Diagnostics & Evaluations

Evaluate in-domain and out-of-domain performance:

```bash
# In-domain evaluation
python eval_in_domain.py

# Run diagnostics
python diagnose.py
python deep_diagnose.py
```

### 4. Running Unit Tests

Verify pipeline and dataloader integrity:

```bash
pytest tests/
```

---

## 📈 Paper & Overleaf Bundle

The full scientific manuscript, formatted for publication in Elsevier format, is located in the [`paper/`](paper/) directory:
- Includes abstract, introduction, methods, results, and discussion.
- All high-resolution figures (`paper/figures/`).
- Complete zip archive ready for one-click upload to Overleaf: [`paper_ready_for_overleaf.zip`](paper_ready_for_overleaf.zip).

---

## 👥 Authors & Acknowledgments

- **Author**: [@pavan67-git](https://github.com/pavan67-git)
- Data provided by **Central Pollution Control Board (CPCB)**, **Open-Meteo**, and **European Space Agency Sentinel-5P via Google Earth Engine**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
