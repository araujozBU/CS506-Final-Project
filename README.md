# The BU Commuter's Guide to the B-Branch: Predicting Transit Reliability
### *A Segment-Level Analysis of Weather and Academic Schedule Impacts*

[INSERT YOUTUBE VIDEO LINK HERE]

## Project Description

Because the MBTA's Green Line B-Branch runs at street level through Boston University, it is uniquely susceptible to two types of disruptions:

1. **Weather** — Rain and snow that affect visibility and track friction, and subsequently how quickly a car can stop and start.

2. **BU Class Schedules** — BU "class transition windows" that spike boarding volumes at key stations as students hustle to and from classes.

This project builds a context-aware predictive model that treats a commute along the B-line as a sequence of **Running Time** (inter-station movement) and **Dwell Time** (platform boarding) segments rather than a single event. 

Two XGBoost regressors are trained separately on years of data from the MBTA for these two components, then combined into a trip calculator that accounts for how weather, class transition periods, and the BU Academic Calendar affect a student's specific commute.

**Why include the BU Academic Calendar?** The calendar tells the model when school is actually in session. For example, on a Saturday in the first week of January, there are no classes, so the model doesn't need to look for class transition surges that aren't happening. 

**Current model performance:**
- Dwell Time MAE: ~20 seconds
   - When the model predicts how long the train will sit at a stop while people board, it's off by about 20 seconds, on average.
- Running Time MAE: ~28 seconds
   - When the model predicts how long it takes to travel between two stops, it's off by about 28 seconds on average.

The project includes a Streamlit web app where users configure their trip, weather conditions, and BU class context to receive a segment-by-segment commute prediction.

---

## Repository Structure

```
.
├── requirements.txt              # Python dependencies
├── figures/
│   ├── fig1_hourly_class_vs_nonclass.png
│   ├── fig2_weather_impact.png
│   ├── fig3_data_processing.png
│   ├── fig4_stop_pair_times.png
│   ├── fig5_feature_importances.png
│   └── fig6_results.png
├── scripts/
│   ├── dataset_creation.py       # ETL pipeline — downloads raw MBTA data, merges weather, outputs Parquet
│   ├── dataset_example.py        # Utility — streams sample rows from Hugging Face for inspection
│   ├── data_visualization.py     # Generates exploratory figures saved to figures/
│   ├── model.py                  # Model training, backtesting, and prediction logic
│   ├── app.py                    # Streamlit web UI
│   └── model_artifacts/
│       ├── model.pkl             # Trained XGBoost model
│       ├── label_encoder.pkl     # Stop-pair label encoder
│       └── metadata.json         # Stop names, model metrics, feature importance
```

---

## Environment

- **Python:** 3.10 or later
- **OS:** macOS, Linux, or Windows (WSL recommended on Windows)
- **Hardware:** No GPU required; training runs on CPU in under a few minutes
- **Disk:** No large download required to run the app, model artifacts are included. Rebuilding the dataset from scratch (~500 MB) requires running dataset_creation.py separately.

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
- **Weather** — Temperature (°C), precipitation (mm), snow depth (mm)
- **BU Context** — Toggle on/off to reflect if it's a BU class day

The app displays a segment-by-segment breakdown showing predicted time versus the historical baseline for each **stop pair**.

- A "stop pair" is two consecutive stations — for example, "BU East → BU Central." The B-Line trip is broken into a series of these pairs, and the app predicts the travel time for each stop pair individually, rather than giving a single time prediction for the whole trip. 

- Why? Delays are not evenly distributed: a class transition surge might slow boarding at BU Central while the rest of the route is unaffected. Predicting each segment separately lets the model capture these localized effects instead of averaging them away.

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

- Why? This won't allow the model to see future data in training, and mimics the way things would work in real life: analyzing past T transit times and evaluating on future data.

To run backtesting, uncomment the `backtest(df)` call at the bottom of `scripts/model.py` and re-run:

```bash
python scripts/model.py
```

This performs a 4-split temporal cross-validation using `GroupShuffleSplit` by stop pair, printing MAE and R² for each split.

- The dataset is divided into 4 time windows. The model trains on window 1, tests on window 2, then trains on windows 1–2, tests on window 3 — and so on. Each split produces a score, and then those scores are averaged to judge overall accuracy.

   - Why? This avoids overfitting.

- The splitting ensures that data from the same pair of stations (i.e., all "BU East → BU Central" rows) won't be split across train and test sets.

   - Why? This could otherwise inflate the model's appearance of accuracy.

- MAE (Mean Absolute Error) — how many seconds off the model's predictions are, on average.
   - It is directly interpretable: an MAE of 20 seconds means the prediction is wrong by about 20 seconds.

- R² (R-squared) — how much of the real-world variation in travel times the model explains, on a 0–1 scale (1 = perfect).
   - MAE alone doesn't tell if the model is actually capturing patterns or just guessing near the average every time. R² reveals whether the model responds meaningfully to changing conditions.

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
