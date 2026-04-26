# The BU Commuter's Guide to the B-Branch: Predicting Transit Reliability
### *A Segment-Level Analysis of Weather and Academic Schedule Impacts*

## Project Description

The MBTA Green Line B-Branch serves as the primary transit corridor for Boston University students along Commonwealth Avenue. Because it runs at street level, it is uniquely susceptible to two types of disruptions:

1. **Meteorological Shocks** — Rain and snow affecting track friction, visibility, and portal entry speeds.
2. **Social Shocks** — "Student Surges" during BU class transition windows that spike boarding volumes at key stations.

This project builds a context-aware predictive model that treats a commute as a sequence of **Running Time** (inter-station movement) and **Dwell Time** (platform boarding) segments rather than a single event. Two XGBoost regressors are trained separately on these components, then composed into a trip calculator that accounts for how weather and the BU Academic Calendar affect a student's specific commute.

**Current model performance:**
- Dwell Time MAE: ~20 seconds
- Running Time MAE: ~28 seconds

The project exposes a Streamlit web app where users configure their trip, weather conditions, and BU context to receive a segment-by-segment commute prediction.

---

## Repository Structure

```
.
├── requirements.txt              # Python dependencies
├── scripts/
│   ├── dataset_creation.py       # ETL pipeline — downloads raw MBTA data, merges weather, outputs Parquet
│   ├── dataset_example.py        # Utility — streams sample rows from Hugging Face for inspection
│   ├── model.py                  # Model training, backtesting, and prediction logic
│   ├── app.py                    # Streamlit web UI
│   └── model_artifacts/
│       ├── model_dwell.pkl       # Trained XGBoost dwell-time model
│       ├── model_travel.pkl      # Trained XGBoost running-time model
│       ├── label_encoder.pkl     # Stop-pair label encoder
│       └── metadata.json         # Stop names, model metrics, feature importance
```

---

## Environment

- **Python:** 3.10 or later
- **OS:** macOS, Linux, or Windows (WSL recommended on Windows)
- **Hardware:** No GPU required; training runs on CPU in under a few minutes
- **Disk:** ~500 MB for raw MBTA data download and Parquet output

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/araujozBU/CS506-Final-Project.git
cd CS506-Final-Project

# 2. Install all dependencies
make install

# 3. Launch the web app
make run
```

Open the URL printed in your terminal (typically `http://localhost:8501`). Configure:

- **Route** — Start and end station (8 stations across Westbound/Eastbound)
- **Time** — Hour, day of week, month
- **Weather** — Temperature (°F), precipitation (mm), snow depth (mm)
- **BU context** — Toggle "BU Class Day" to auto-infer active hours and surge flags

The app displays a segment-by-segment breakdown showing predicted time versus the historical baseline for each stop pair.

### Inspect a Dataset Sample

```bash
python scripts/dataset_example.py
```

Streams 10 rows from the Hugging Face ML-ready dataset for quick inspection without a local download.

---

## Data Sources

| Source | Description |
|--------|-------------|
| [MBTA Rapid Transit Travel Times](https://mbta-massdot.opendata.arcgis.com/datasets/5f71a5c035fc4a4dad1b7fa73ba27ef8/) | Raw arrival/departure event data for the B-Branch (2024–2026), hosted on Hugging Face as `adybacki/24_25_26_mbta_lr_travel_times` |
| [Meteostat](https://meteostat.net/) | Hourly weather data for Boston Logan (station ID: `72503`) |
| BU Academic Calendar | Manually curated semester ranges, holidays, spring breaks, and class transition windows for 2024–2026 |
| ML-Ready Dataset | Processed and feature-engineered dataset on Hugging Face: `adybacki/bu_green_line_ml_ready` |

---

## Testing

The project includes a built-in **walk-forward backtesting** function in [`scripts/model.py`](scripts/model.py) that evaluates model generalization by training on earlier data and testing on later dates.

To run backtesting, uncomment the `backtest(df)` call at the bottom of `scripts/model.py` and re-run:

```bash
python scripts/model.py
```

This performs a 4-split temporal cross-validation using `GroupShuffleSplit` by stop pair, printing MAE and R² for each split.

To verify the dataset pipeline, `dataset_example.py` streams a small sample from Hugging Face for manual inspection:

```bash
python scripts/dataset_example.py
```

---

## Contributing

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes.** Keep each branch focused on a single concern (data pipeline, model changes, UI, etc.).

3. **Test your changes** by running the backtesting function and verifying the Streamlit app loads without errors.

4. **Commit** with a clear message describing the *why* behind the change.

5. **Open a pull request** against `main`. Include:
   - A short description of what changed and why
   - Any impact on model metrics (MAE, R²) if model artifacts were retrained

**Reporting bugs or ideas:** Open an issue on the [GitHub repository](https://github.com/araujozBU/CS506-Final-Project/issues) with as much context as possible (OS, Python version, error message, steps to reproduce).

---

## Team

| Name | Email |
|------|-------|
| Zaki Araujo | araujoz@bu.edu |
| Adrian Dybacki | adybacki@bu.edu |
| Andrew Botolino | botolino@bu.edu |
| Kuba Rozwadowski | kubaroz@bu.edu |
| Rohan Chablani | rohan204@bu.edu |

**Repository:** [https://github.com/araujozBU/CS506-Final-Project](https://github.com/araujozBU/CS506-Final-Project)
