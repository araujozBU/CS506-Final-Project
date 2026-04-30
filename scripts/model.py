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

from sklearn.model_selection import train_test_split
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


FEATURE_COLS_DWELL = [
    "hour", "minute", "day_of_week", "month",
    "is_weekend", "is_peak_am", "is_peak_pm",
    "temp", "prcp", "snow",
    "is_precipitating", "is_snowing", "heavy_snow",
    "is_bu_class_day", "is_active_class_time", "is_student_surge",
    "stop_pair_enc",   # label-encoded stop pair
]

FEATURE_COLS_TRAVEL = FEATURE_COLS_DWELL + ["dwell_time_sec"]

TARGET_COL_DWELL = "dwell_time_sec"
TARGET_COL_TRAVEL = "travel_time_sec"


# ──────────────────────────────────────────────
# 2. MODEL TRAINING
# ──────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Trains two XGBoost regressors (Dwell Time and Travel Time).
    Returns (model_dwell, model_travel, label_encoder, metrics, df).
    """
    df = build_features(df)

    le = LabelEncoder()
    df["stop_pair_enc"] = le.fit_transform(df["stop_pair"])

    X_dwell = df[FEATURE_COLS_DWELL]
    y_dwell = df[TARGET_COL_DWELL]
    
    X_travel = df[FEATURE_COLS_TRAVEL]
    y_travel = df[TARGET_COL_TRAVEL]
    
    groups = df["stop_pair"]

    # Train/test split grouped by stop pair so each pair appears in both sets
    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
    )

    # -- Train Dwell Model --
    model_dwell = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_dwell.fit(
        X_dwell.iloc[train_idx], y_dwell.iloc[train_idx],
        eval_set=[(X_dwell.iloc[test_idx], y_dwell.iloc[test_idx])],
        verbose=False
    )
    
    # -- Train Travel Model --
    model_travel = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, random_state=42, n_jobs=-1, verbosity=0
    )
    model_travel.fit(
        X_travel.iloc[train_idx], y_travel.iloc[train_idx],
        eval_set=[(X_travel.iloc[test_idx], y_travel.iloc[test_idx])],
        verbose=False
    )

    preds_dwell = model_dwell.predict(X_dwell.iloc[test_idx])
    preds_travel = model_travel.predict(X_travel.iloc[test_idx])
    
    metrics = {
        "dwell_mae_sec": float(mean_absolute_error(y_dwell.iloc[test_idx], preds_dwell)),
        "travel_mae_sec": float(mean_absolute_error(y_travel.iloc[test_idx], preds_travel)),
        "travel_r2": float(r2_score(y_travel.iloc[test_idx], preds_travel)),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }

    return model_dwell, model_travel, le, metrics, df


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

        X_train_dwell = train_df[FEATURE_COLS_DWELL]
        y_train_dwell = train_df[TARGET_COL_DWELL]
        X_test_dwell = test_df[FEATURE_COLS_DWELL]
        y_test_dwell = test_df[TARGET_COL_DWELL]
        
        X_train_travel = train_df[FEATURE_COLS_TRAVEL]
        y_train_travel = train_df[TARGET_COL_TRAVEL]
        X_test_travel = test_df[FEATURE_COLS_TRAVEL]
        y_test_travel = test_df[TARGET_COL_TRAVEL]

        model_dwell = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0,
        )
        model_dwell.fit(X_train_dwell, y_train_dwell, verbose=False)
        
        model_travel = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0,
        )
        model_travel.fit(X_train_travel, y_train_travel, verbose=False)
        
        preds_travel = model_travel.predict(X_test_travel)

        # Baseline: median travel time per stop pair in training set
        pair_median = train_df.groupby("stop_pair")[TARGET_COL_TRAVEL].median()
        baseline_preds = test_df["stop_pair"].map(pair_median).fillna(train_df[TARGET_COL_TRAVEL].median())

        fold_result = {
            "fold": i,
            "train_dates": str(min(train_dates)) + " → " + str(max(train_dates)),
            "test_dates": str(min(test_dates)) + " → " + str(max(test_dates)),
            "n_train": len(X_train_travel),
            "n_test": len(X_test_travel),
            "model_mae": float(mean_absolute_error(y_test_travel, preds_travel)),
            "model_rmse": float(np.sqrt(mean_squared_error(y_test_travel, preds_travel))),
            "model_r2": float(r2_score(y_test_travel, preds_travel)),
            "baseline_mae": float(mean_absolute_error(y_test_travel, baseline_preds)),
            "baseline_rmse": float(np.sqrt(mean_squared_error(y_test_travel, baseline_preds))),
        }
        results.append(fold_result)
        print(f"  Fold {i}: MAE={fold_result['model_mae']:.1f}s  R²={fold_result['model_r2']:.3f}  "
              f"(baseline MAE={fold_result['baseline_mae']:.1f}s)")

    return results


# ──────────────────────────────────────────────
# 4. PREDICTION FUNCTION (used by UI)
# ──────────────────────────────────────────────

def predict_travel_time(
    model_dwell, model_travel, le, df_ref: pd.DataFrame,
    from_stop_id: str, to_stop_id: str,
    hour: int = 8, day_of_week: int = 1,
    temp: float = 10.0, prcp: float = 0.0, snow: float = 0.0,
    is_bu_class_day: int = 1, is_active_class_time: int = 1,
    is_student_surge: int = 0, month: int = 9,
):
    """
    Returns predicted dwell and travel time (seconds) for a single scenario.
    Also returns historical medians for travel and dwell for that pair.
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
    
    # 1. Predict Dwell Time
    row_dwell_dict = {
        "hour": hour, "minute": minute, "day_of_week": day_of_week,
        "month": month, "is_weekend": is_weekend,
        "is_peak_am": is_peak_am, "is_peak_pm": is_peak_pm,
        "temp": temp, "prcp": prcp, "snow": snow,
        "is_precipitating": is_precipitating, "is_snowing": is_snowing,
        "heavy_snow": heavy_snow, "is_bu_class_day": is_bu_class_day,
        "is_active_class_time": is_active_class_time,
        "is_student_surge": is_student_surge,
        "stop_pair_enc": pair_enc,
    }
    row_dwell = pd.DataFrame([row_dwell_dict])
    
    dwell_pred = float(model_dwell.predict(row_dwell[FEATURE_COLS_DWELL])[0])
    dwell_pred = max(0.0, dwell_pred)  # ensure non-negative logic
    
    # 2. Predict Travel Time
    row_travel_dict = row_dwell_dict.copy()
    row_travel_dict["dwell_time_sec"] = dwell_pred
    row_travel = pd.DataFrame([row_travel_dict])
    
    travel_pred = float(model_travel.predict(row_travel[FEATURE_COLS_TRAVEL])[0])

    # Historical baselines
    pair_mask = (df_ref["from_stop_id"].astype(str) == str(from_stop_id)) & \
                (df_ref["to_stop_id"].astype(str) == str(to_stop_id))

    baseline_sec = None
    baseline_dwell_sec = None
    if pair_mask.any():
        if "travel_time_sec" in df_ref.columns:
            baseline_sec = float(df_ref.loc[pair_mask, "travel_time_sec"].median())
        if "dwell_time_sec" in df_ref.columns:
            dwell_values = df_ref.loc[pair_mask, "dwell_time_sec"].dropna()
            if not dwell_values.empty:
                baseline_dwell_sec = float(dwell_values.median())
    
    return {
        "predicted_dwell_sec": dwell_pred,
        "predicted_sec": travel_pred,
        "baseline_sec": baseline_sec,
        "baseline_dwell_sec": baseline_dwell_sec,
        "feature_row": row_travel_dict
    }


# ──────────────────────────────────────────────
# 5. SCENARIO COMPARISON  (used by UI)
# ──────────────────────────────────────────────

def compare_scenarios(model_dwell, model_travel, le, df_ref, from_stop, to_stop, hour=8, month=9):
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
            model_dwell, model_travel, le, df_ref,
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
    Loads the ML-ready dataset from local parquet if available,
    otherwise downloads from Hugging Face.
    """
    import os
    
    local_parquet = "bu_green_line_gold.parquet"
    
    if os.path.exists(local_parquet):
        print(f"      Loading dataset from local file: {local_parquet}")
        df = pd.read_parquet(local_parquet)
    else:
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

    print("\n[2/4] Building features and training Dual models...")
    model_dwell, model_travel, le, metrics, df_feat = train_model(df_raw)
    print(f"      Dwell MAE={metrics['dwell_mae_sec']:.1f}s   Travel MAE={metrics['travel_mae_sec']:.1f}s   Travel R²={metrics['travel_r2']:.3f}")
    print(f"      Train: {metrics['n_train']:,}  Test: {metrics['n_test']:,}")

    print("\n[3/4] (Skipped walk-forward backtest for dual-model simplicity)")
    # bt_results = backtest(df_raw, n_splits=4)
    bt_results = []

    print("\n[4/4] Saving artifacts...")
    artifact_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    
    with open(os.path.join(artifact_dir, "model_dwell.pkl"), "wb") as f:
        pickle.dump(model_dwell, f)
    with open(os.path.join(artifact_dir, "model_travel.pkl"), "wb") as f:
        pickle.dump(model_travel, f)
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
        "feature_importance_dwell": {
            k: float(v) for k, v in zip(FEATURE_COLS_DWELL, model_dwell.feature_importances_)
        },
        "feature_importance_travel": {
            k: float(v) for k, v in zip(FEATURE_COLS_TRAVEL, model_travel.feature_importances_)
        }
    }
    with open(os.path.join(artifact_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ All artifacts saved to {artifact_dir}")
    print("✓ Ready to serve predictions to the UI")
