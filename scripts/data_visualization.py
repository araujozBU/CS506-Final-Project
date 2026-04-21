"""
BU Green Line — April Check-In Visualizations
==============================================
6 clean figures covering all rubric sections.

Run:
    python3 scripts/data_visualization.py

Outputs saved to ./figures/
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── CONFIG ─────────────────────────────────────────────────────────────────────

PARQUET_PATH = "/home/rohan/VSCODE/FinalProject506/CS506-Final-Project/bu_green_line_gold.parquet"
OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

STOP_NAMES = {
    "70153": "Hynes",      "71151": "Kenmore",    "70149": "Blandford St",
    "70147": "BU East",    "70145": "BU Central", "170141": "Amory St",
    "170137": "Babcock St","70134": "Packard's",  "170136": "Babcock St",
    "170140": "Amory St",  "70144": "BU Central", "70146": "BU East",
    "70148": "Blandford St","71150": "Kenmore",
}

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ── LOAD DATA ──────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_parquet(PARQUET_PATH)

df["travel_time_sec"] = (
    df["to_stop_arrival_datetime"] - df["from_stop_departure_datetime"]
).dt.total_seconds()

# Keep only realistic travel times and service hours (6am–11pm)
df = df[(df["travel_time_sec"] > 0) & (df["travel_time_sec"] < 600)]
df = df[(df["hour"] >= 6) & (df["hour"] <= 23)]

df["stop_pair_name"] = (
    df["from_stop_id"].map(STOP_NAMES).fillna(df["from_stop_id"]) + " → " +
    df["to_stop_id"].map(STOP_NAMES).fillna(df["to_stop_id"])
)
df["snow"]  = df["snow"].fillna(0)
df["prcp"]  = df["prcp"].fillna(0)
df["temp"]  = df["temp"].fillna(df["temp"].median())
df["dwell_time_sec"] = df["dwell_time_sec"].fillna(0).clip(0, 600)

print(f"  {len(df):,} rows ready\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Travel Time by Hour: Class Day vs Non-Class Day
# ══════════════════════════════════════════════════════════════════════════════
print("[1/6] Hourly travel time: class day vs non-class day")

hourly = (
    df.groupby(["hour", "is_bu_class_day"])["travel_time_sec"]
    .median()
    .reset_index()
)
class_df   = hourly[hourly["is_bu_class_day"] == 1]
noclass_df = hourly[hourly["is_bu_class_day"] == 0]

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(class_df["hour"],   class_df["travel_time_sec"],
        color="#CC0000", lw=2.5, marker="o", ms=5, label="BU Class Day")
ax.plot(noclass_df["hour"], noclass_df["travel_time_sec"],
        color="#00843D", lw=2.5, marker="o", ms=5, label="Non-Class Day")

for xstart, xend, label in [(8, 10, "Morning Peak"), (12, 14, "Midday Peak"), (16, 18, "Afternoon Peak")]:
    ax.axvspan(xstart, xend, alpha=0.10, color="#F5A623")
    mid = (xstart + xend) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 1 else 95,
            label, ha="center", va="top", fontsize=8, color="#b07000")

ax.set_xticks(range(6, 24))
ax.set_xticklabels([f"{h}:00" for h in range(6, 24)], rotation=45, fontsize=8)
ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Median Travel Time (seconds)", fontsize=11)
ax.set_title("Green Line Travel Time by Hour — Class Days vs. Non-Class Days",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4, linestyle="--")
fig.tight_layout()
save(fig, "fig1_hourly_class_vs_nonclass.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Weather Impact on Travel Time
# ══════════════════════════════════════════════════════════════════════════════
print("[2/6] Weather impact")

df["weather_cat"] = "Clear"
df.loc[df["prcp"] > 0, "weather_cat"] = "Rain"
df.loc[df["snow"] > 0, "weather_cat"] = "Snow"
df.loc[df["snow"] > 5, "weather_cat"] = "Heavy Snow"

cat_order  = ["Clear", "Rain", "Snow", "Heavy Snow"]
cat_colors = ["#00843D", "#4A9EFF", "#90C0F0", "#C0D8FF"]

stats = (
    df.groupby("weather_cat")["travel_time_sec"]
    .agg(median="median", q25=lambda x: x.quantile(0.25),
         q75=lambda x: x.quantile(0.75), count="count")
    .reindex(cat_order)
)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(cat_order, stats["median"], color=cat_colors,
              edgecolor="#333", width=0.55, zorder=3)
ax.errorbar(cat_order, stats["median"],
            yerr=[stats["median"] - stats["q25"], stats["q75"] - stats["median"]],
            fmt="none", color="#333", capsize=6, lw=1.5, zorder=4)

for bar, med, n in zip(bars, stats["median"], stats["count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, med + 2,
            f"{med:.0f}s\n(n={n:,})", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Median Travel Time (seconds)", fontsize=11)
ax.set_xlabel("Weather Condition", fontsize=11)
ax.set_title("Impact of Weather on Green Line Travel Time",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, stats["q75"].max() * 1.35)
ax.grid(axis="y", alpha=0.4, linestyle="--", zorder=0)
ax.text(0.01, -0.15, "Error bars show 25th–75th percentile range.",
        transform=ax.transAxes, fontsize=8, color="#666")
fig.tight_layout()
save(fig, "fig2_weather_impact.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Data Volume Over Time + Missing Values
# ══════════════════════════════════════════════════════════════════════════════
print("[3/6] Data processing overview")

df_raw = pd.read_parquet(PARQUET_PATH)
miss = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
miss = miss[miss > 0]

df_raw["date_only"] = pd.to_datetime(df_raw["date_only"])
monthly = df_raw.groupby(df_raw["date_only"].dt.to_period("M")).size()
month_labels = [str(p) for p in monthly.index]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.bar(range(len(monthly)), monthly.values, color="#00843D", edgecolor="#004d22", alpha=0.85)
step = max(1, len(monthly) // 8)
ax.set_xticks(range(0, len(monthly), step))
ax.set_xticklabels([month_labels[i] for i in range(0, len(monthly), step)],
                   rotation=35, fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylabel("Trip Segments Recorded", fontsize=11)
ax.set_title("Data Volume Over Time", fontsize=12, fontweight="bold", pad=10)
ax.grid(axis="y", alpha=0.4, linestyle="--")

ax2 = axes[1]
if len(miss) == 0:
    ax2.text(0.5, 0.5, "No missing values!", ha="center", va="center",
             fontsize=13, transform=ax2.transAxes)
else:
    colors = ["#CC0000" if v > 10 else "#F5A623" if v > 1 else "#aaaaaa"
              for v in miss.values]
    ax2.barh(miss.index, miss.values, color=colors, edgecolor="#333", height=0.6)
    for i, v in enumerate(miss.values):
        ax2.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax2.set_xlabel("% Missing", fontsize=11)
ax2.set_title("Missing Values by Column", fontsize=12, fontweight="bold", pad=10)
ax2.grid(axis="x", alpha=0.4, linestyle="--")

fig.tight_layout(pad=2.5)
save(fig, "fig3_data_processing.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Average Travel Time Per Stop Pair
# ══════════════════════════════════════════════════════════════════════════════
print("[4/6] Travel time per stop pair")

pair_stats = (
    df.groupby("stop_pair_name")["travel_time_sec"]
    .agg(median="median", count="count")
    .sort_values("median", ascending=True)
)
pair_stats = pair_stats[pair_stats["count"] > 500]

fig, ax = plt.subplots(figsize=(10, max(4, len(pair_stats) * 0.45)))
bars = ax.barh(pair_stats.index, pair_stats["median"],
               color="#00843D", edgecolor="#004d22", height=0.6)
for bar, med in zip(bars, pair_stats["median"]):
    ax.text(med + 1, bar.get_y() + bar.get_height() / 2,
            f"{med:.0f}s", va="center", fontsize=9)

ax.set_xlabel("Median Travel Time (seconds)", fontsize=11)
ax.set_title("Median Travel Time Per Stop-to-Stop Segment",
             fontsize=13, fontweight="bold", pad=12)
ax.grid(axis="x", alpha=0.4, linestyle="--")
fig.tight_layout()
save(fig, "fig4_stop_pair_times.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Feature Importances
# ══════════════════════════════════════════════════════════════════════════════
print("[5/6] Feature importances")

sample = df.sample(min(60_000, len(df)), random_state=42).copy()
le = LabelEncoder()
sample["pair_enc"] = le.fit_transform(
    sample["from_stop_id"].astype(str) + "_" + sample["to_stop_id"].astype(str)
)
sample["is_peak_am"] = ((sample["hour"] >= 7) & (sample["hour"] <= 9)).astype(int)
sample["is_peak_pm"] = ((sample["hour"] >= 16) & (sample["hour"] <= 19)).astype(int)
sample["is_weekend"] = (sample["day_of_week"] >= 5).astype(int)

FEATS = ["hour", "day_of_week", "is_weekend", "is_peak_am", "is_peak_pm",
         "temp", "prcp", "snow",
         "is_bu_class_day", "is_active_class_time", "is_student_surge",
         "dwell_time_sec", "pair_enc"]
FEAT_LABELS = ["Hour of Day", "Day of Week", "Is Weekend", "AM Peak (7–9am)",
               "PM Peak (4–7pm)", "Temperature", "Precipitation", "Snowfall",
               "BU Class Day", "Active Class Period", "Student Surge",
               "Dwell Time at Stop", "Stop Pair"]

X = sample[FEATS]
y = sample["travel_time_sec"]

model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                          subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
model.fit(X, y)

imp = pd.Series(model.feature_importances_, index=FEAT_LABELS).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#CC0000" if v >= imp.nlargest(3).min() else "#00843D" for v in imp.values]
ax.barh(imp.index, imp.values, color=colors, edgecolor="#333", height=0.65)
for i, v in enumerate(imp.values):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

ax.set_xlabel("Feature Importance Score", fontsize=11)
ax.set_title("What Factors Most Affect Green Line Travel Time?\n(XGBoost Feature Importances)",
             fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="x", alpha=0.4, linestyle="--")

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#CC0000", label="Top 3 factors"),
                   Patch(color="#00843D", label="Other factors")],
          fontsize=9, loc="lower right")
fig.tight_layout()
save(fig, "fig5_feature_importances.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Predicted vs Actual + Residuals
# ══════════════════════════════════════════════════════════════════════════════
print("[6/6] Predicted vs Actual")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                     subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
m.fit(X_train, y_train)
y_pred = m.predict(X_test)
residuals = y_test.values - y_pred

pair_med = sample.loc[X_train.index].groupby("pair_enc")["travel_time_sec"].median()
baseline = sample.loc[X_test.index, "pair_enc"].map(pair_med).fillna(y_train.median())
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)
b_mae = mean_absolute_error(y_test, baseline)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
lim = (0, 400)
ax.scatter(y_test, y_pred, alpha=0.07, s=6, color="#00843D", rasterized=True)
ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual Travel Time (seconds)", fontsize=11)
ax.set_ylabel("Predicted Travel Time (seconds)", fontsize=11)
ax.set_title("Predicted vs. Actual Travel Time", fontsize=12, fontweight="bold", pad=10)
ax.text(0.04, 0.95,
        f"MAE = {mae:.1f}s\nR² = {r2:.3f}\nBaseline MAE = {b_mae:.1f}s",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.4"))
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")

ax2 = axes[1]
ax2.hist(residuals, bins=60, color="#00843D", edgecolor="white", alpha=0.8)
ax2.axvline(0, color="#CC0000", lw=2, label="Zero error")
ax2.axvline(residuals.mean(), color="#F5A623", lw=1.5, ls="--",
            label=f"Mean error = {residuals.mean():.1f}s")
ax2.set_xlabel("Prediction Error (Actual − Predicted, seconds)", fontsize=11)
ax2.set_ylabel("Number of Trips", fontsize=11)
ax2.set_title("Distribution of Prediction Errors", fontsize=12, fontweight="bold", pad=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3, linestyle="--")

fig.tight_layout(pad=2)
save(fig, "fig6_results.png")

# ── DONE ───────────────────────────────────────────────────────────────────────

print(f"""
Done! Figures saved to {OUT_DIR}/

  fig1 — Hourly travel time: class day vs non-class day   [Data Viz]
  fig2 — Weather impact on travel time                    [Data Viz]
  fig3 — Data volume over time + missing values           [Data Processing]
  fig4 — Median travel time per stop segment              [Data Processing]
  fig5 — Feature importances                              [Modeling]
  fig6 — Predicted vs actual + error distribution         [Results]

Model performance:
  MAE      = {mae:.1f}s
  R2       = {r2:.3f}
  Baseline = {b_mae:.1f}s MAE
""")