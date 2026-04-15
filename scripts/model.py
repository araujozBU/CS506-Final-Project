"""
BU Green Line Transit Time Prediction Model
============================================
Trains an XGBoost model to predict travel time between stop pairs,
with features: weather (temp, prcp, snow), BU class day flags,
hour of day, day of week, and dwell time.

Also includes:
  - backtesting (walk-forward validation by date)
  - a prediction function used by the UI
  - baseline comparison (median by stop-pair + time context)
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ──────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and return a clean feature matrix."""
    df = df.copy()

    # Parse datetimes if they're strings
    for col in ["from_stop_departure_datetime", "to_stop_arrival_datetime"]:
        if df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], utc=True)

    # Target: actual travel time in seconds
    df["travel_time_sec"] = (
        df["to_stop_arrival_datetime"] - df["from_stop_departure_datetime"]
    ).dt.total_seconds()

    # Drop bad rows
    df = df[df["travel_time_sec"] > 0]
    df = df[df["travel_time_sec"] < 3600]  # cap at 1 hour (outlier filter)

    # Stop-pair identifier
    df["stop_pair"] = df["from_stop_id"].astype(str) + "_" + df["to_stop_id"].astype(str)

    # Time features
    df["hour"] = df["from_stop_departure_datetime"].dt.hour
    df["minute"] = df["from_stop_departure_datetime"].dt.minute
    df["day_of_week"] = df["from_stop_departure_datetime"].dt.dayofweek
    df["month"] = df["from_stop_departure_datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_am"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
    df["is_peak_pm"] = ((df["hour"] >= 16) & (df["hour"] <= 19)).astype(int)

    # Weather: fill NaN with 0
    for col in ["temp", "prcp", "snow"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    df["is_precipitating"] = (df["prcp"] > 0).astype(int)
    df["is_snowing"] = (df["snow"] > 0).astype(int)
    df["heavy_snow"] = (df["snow"] > 5).astype(int)

    # BU flags
    for col in ["is_bu_class_day", "is_active_class_time", "is_student_surge"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    # Dwell time at origin
    df["dwell_time_sec"] = df["dwell_time_sec"].fillna(0).clip(0, 600)

    return df


FEATURE_COLS = [
    "hour", "minute", "day_of_week", "month",
    "is_weekend", "is_peak_am", "is_peak_pm",
    "temp", "prcp", "snow",
    "is_precipitating", "is_snowing", "heavy_snow",
    "is_bu_class_day", "is_active_class_time", "is_student_surge",
    "dwell_time_sec",
    "stop_pair_enc",   # label-encoded stop pair
]
TARGET_COL = "travel_time_sec"


# ──────────────────────────────────────────────
# 2. MODEL TRAINING
# ──────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Trains an XGBoost regressor. Returns (model, label_encoder, feature_cols).
    """
    df = build_features(df)

    le = LabelEncoder()
    df["stop_pair_enc"] = le.fit_transform(df["stop_pair"])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    groups = df["stop_pair"]  # group by stop pair for split

    # Train/test split grouped by stop pair so each pair appears in both sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    metrics = {
        "mae_sec": float(mean_absolute_error(y_test, preds)),
        "rmse_sec": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return model, le, metrics, df


# ──────────────────────────────────────────────
# 3. BACKTESTING (walk-forward by date)
# ──────────────────────────────────────────────

def backtest(df: pd.DataFrame, n_splits: int = 4):
    """
    Walk-forward backtest: train on earlier dates, test on later dates.
    Returns a list of dicts with metrics per fold.
    """
    df = build_features(df)
    df["date_only"] = pd.to_datetime(df["from_stop_departure_datetime"]).dt.date
    dates = sorted(df["date_only"].unique())
    fold_size = len(dates) // (n_splits + 1)

    results = []
    for i in range(1, n_splits + 1):
        cutoff_idx = i * fold_size
        train_dates = set(dates[:cutoff_idx])
        test_dates = set(dates[cutoff_idx: cutoff_idx + fold_size])

        train_df = df[df["date_only"].isin(train_dates)]
        test_df = df[df["date_only"].isin(test_dates)]

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        le = LabelEncoder()
        # Fit on union so no unseen labels
        all_pairs = pd.concat([train_df["stop_pair"], test_df["stop_pair"]])
        le.fit(all_pairs)
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["stop_pair_enc"] = le.transform(train_df["stop_pair"])
        test_df["stop_pair_enc"] = le.transform(test_df["stop_pair"])

        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)

        # Baseline: median travel time per stop pair in training set
        pair_median = train_df.groupby("stop_pair")[TARGET_COL].median()
        baseline_preds = test_df["stop_pair"].map(pair_median).fillna(train_df[TARGET_COL].median())

        fold_result = {
            "fold": i,
            "train_dates": str(min(train_dates)) + " → " + str(max(train_dates)),
            "test_dates": str(min(test_dates)) + " → " + str(max(test_dates)),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "model_mae": float(mean_absolute_error(y_test, preds)),
            "model_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "model_r2": float(r2_score(y_test, preds)),
            "baseline_mae": float(mean_absolute_error(y_test, baseline_preds)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_preds))),
        }
        results.append(fold_result)
        print(f"  Fold {i}: MAE={fold_result['model_mae']:.1f}s  R²={fold_result['model_r2']:.3f}  "
              f"(baseline MAE={fold_result['baseline_mae']:.1f}s)")

    return results


# ──────────────────────────────────────────────
# 4. PREDICTION FUNCTION (used by UI)
# ──────────────────────────────────────────────

def predict_travel_time(
    model, le, df_ref: pd.DataFrame,
    from_stop_id: str, to_stop_id: str,
    hour: int = 8, day_of_week: int = 1,
    temp: float = 10.0, prcp: float = 0.0, snow: float = 0.0,
    is_bu_class_day: int = 1, is_active_class_time: int = 1,
    is_student_surge: int = 0, month: int = 9,
):
    """
    Returns predicted travel time (seconds) for a single scenario.
    Also returns 'baseline' (historical median for that pair + hour bucket).
    """
    stop_pair = f"{from_stop_id}_{to_stop_id}"

    # Encode stop pair (handle unseen pairs)
    try:
        pair_enc = le.transform([stop_pair])[0]
    except ValueError:
        pair_enc = 0  # fallback

    minute = 0
    is_weekend = int(day_of_week >= 5)
    is_peak_am = int(7 <= hour <= 9)
    is_peak_pm = int(16 <= hour <= 19)
    is_precipitating = int(prcp > 0)
    is_snowing = int(snow > 0)
    heavy_snow = int(snow > 5)

    # Typical dwell time for this pair from training data
    pair_mask = (df_ref["from_stop_id"].astype(str) == str(from_stop_id)) & \
                (df_ref["to_stop_id"].astype(str) == str(to_stop_id))
    dwell = float(df_ref.loc[pair_mask, "dwell_time_sec"].median()) if pair_mask.any() else 30.0

    row = pd.DataFrame([{
        "hour": hour, "minute": minute, "day_of_week": day_of_week,
        "month": month, "is_weekend": is_weekend,
        "is_peak_am": is_peak_am, "is_peak_pm": is_peak_pm,
        "temp": temp, "prcp": prcp, "snow": snow,
        "is_precipitating": is_precipitating, "is_snowing": is_snowing,
        "heavy_snow": heavy_snow, "is_bu_class_day": is_bu_class_day,
        "is_active_class_time": is_active_class_time,
        "is_student_surge": is_student_surge,
        "dwell_time_sec": dwell,
        "stop_pair_enc": pair_enc,
    }])

    pred = float(model.predict(row[FEATURE_COLS])[0])

    # Historical baseline
    baseline = None
    if pair_mask.any():
        baseline = float(df_ref.loc[pair_mask, "travel_time_sec"].median()
                         if "travel_time_sec" in df_ref.columns
                         else pred)

    return {"predicted_sec": round(pred, 1), "baseline_sec": baseline}


# ──────────────────────────────────────────────
# 5. SCENARIO COMPARISON  (used by UI)
# ──────────────────────────────────────────────

def compare_scenarios(model, le, df_ref, from_stop, to_stop, hour=8, month=9):
    """
    Return predictions across 4 key scenario combos for a given stop pair.
    Used to generate the 'impact' view in the UI.
    """
    scenarios = {
        "Normal (clear, no class)": dict(temp=15, prcp=0, snow=0, is_bu_class_day=0, is_active_class_time=0, is_student_surge=0),
        "BU Class Day (clear)":     dict(temp=15, prcp=0, snow=0, is_bu_class_day=1, is_active_class_time=1, is_student_surge=0),
        "BU Class + Surge":         dict(temp=15, prcp=0, snow=0, is_bu_class_day=1, is_active_class_time=1, is_student_surge=1),
        "Rain + BU Class":          dict(temp=10, prcp=5, snow=0, is_bu_class_day=1, is_active_class_time=1, is_student_surge=0),
        "Heavy Snow + BU Class":    dict(temp=-2, prcp=0, snow=15, is_bu_class_day=1, is_active_class_time=1, is_student_surge=1),
        "Heavy Snow, No Class":     dict(temp=-2, prcp=0, snow=15, is_bu_class_day=0, is_active_class_time=0, is_student_surge=0),
    }
    results = {}
    for name, kwargs in scenarios.items():
        r = predict_travel_time(
            model, le, df_ref,
            from_stop, to_stop,
            hour=hour, day_of_week=1, month=month, **kwargs
        )
        results[name] = r["predicted_sec"]
    return results


# ──────────────────────────────────────────────
# 6. MAIN — train, backtest, save artifacts
# ──────────────────────────────────────────────

def load_training_data_from_huggingface():
    """
    Loads the real ML-ready dataset from Hugging Face.
    """
    from datasets import load_dataset
    print("      Downloading dataset from Hugging Face...")
    dataset = load_dataset("adybacki/bu_green_line_ml_ready", split="train")
    df = dataset.to_pandas()

    stop_names = {
        # Westbound (Outbound)
        "70153": "Hynes", "71151": "Kenmore", "70149": "Blandford St",
        "70147": "BU East", "70145": "BU Central", "170141": "Amory", "170137": "Babcock",
        # Eastbound (Inbound)
        "70134": "Packard's", "170136": "Babcock", "170140": "Amory",
        "70144": "BU Central", "70146": "BU East", "70148": "Blandford St", "71150": "Kenmore"
    }

    return df, stop_names



if __name__ == "__main__":
    print("=" * 60)
    print("  BU Green Line ML Model — Training & Backtesting")
    print("=" * 60)

    print("\n[1/4] Loading training data...")
    df_raw, stop_names = load_training_data_from_huggingface()
    print(f"      {len(df_raw):,} rows loaded")

    print("\n[2/4] Building features and training model...")
    model, le, metrics, df_feat = train_model(df_raw)
    print(f"      MAE={metrics['mae_sec']:.1f}s  RMSE={metrics['rmse_sec']:.1f}s  R²={metrics['r2']:.3f}")
    print(f"      Train: {metrics['n_train']:,}  Test: {metrics['n_test']:,}")

    print("\n[3/4] Running walk-forward backtest...")
    bt_results = backtest(df_raw, n_splits=4)

    print("\n[4/4] Saving artifacts...")
    artifact_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(os.path.join(artifact_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(artifact_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Save metadata for the UI
    stop_pairs = df_feat["stop_pair"].value_counts().index.tolist()
    metadata = {
        "stop_names": stop_names,
        "stop_sequence": list(stop_names.keys()),
        "stop_pairs": stop_pairs[:200],
        "train_metrics": metrics,
        "backtest_results": bt_results,
        "feature_importance": {
            k: float(v) for k, v in
            zip(FEATURE_COLS, model.feature_importances_)
        }
    }
    with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Quick sanity check predictions
    print("\n  Sample predictions (BU East → BU Central, 8am):")
    for scenario, kwargs in [
        ("Clear, no class", dict(temp=15, prcp=0, snow=0, is_bu_class_day=0, is_active_class_time=0, is_student_surge=0)),
        ("Clear, BU class", dict(temp=15, prcp=0, snow=0, is_bu_class_day=1, is_active_class_time=1, is_student_surge=0)),
        ("Snow 10cm, BU class+surge", dict(temp=-2, prcp=0, snow=10, is_bu_class_day=1, is_active_class_time=1, is_student_surge=1)),
    ]:
        r = predict_travel_time(model, le, df_feat, "70147", "70145", hour=8, **kwargs)
        print(f"    {scenario}: {r['predicted_sec']:.0f}s")

    print(f"\n✓ All artifacts saved to {artifact_dir}")
    print("✓ Ready to serve predictions to the UI")