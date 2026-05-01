"""
Tests for the BU Green Line prediction pipeline.
Covers feature engineering, outlier filtering, BU calendar logic, and surge detection.
"""

import sys
import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

# Allow imports from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# Mock heavy native dependencies so tests can run without them installed
for _mod in ('xgboost', 'meteostat'):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from model import build_features, FEATURE_COLS_DWELL, FEATURE_COLS_TRAVEL
from dataset_creation import get_surge_flag, add_bu_semester_logic


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_raw_df(n=10):
    """Minimal DataFrame that satisfies build_features() input requirements."""
    base = pd.Timestamp("2024-10-15 09:00:00", tz="UTC")
    departures = [base + pd.Timedelta(minutes=i * 5) for i in range(n)]
    arrivals   = [t + pd.Timedelta(seconds=90) for t in departures]
    return pd.DataFrame({
        "from_stop_id":                 ["70147"] * n,
        "to_stop_id":                   ["70145"] * n,
        "trip_id":                      [f"trip_{i}" for i in range(n)],
        "from_stop_departure_datetime": departures,
        "to_stop_arrival_datetime":     arrivals,
        "dwell_time_sec":               [30.0] * n,
        "is_bu_class_day":              [1] * n,
        "is_active_class_time":         [1] * n,
        "is_student_surge":             [0] * n,
        "temp":  [10.0] * n,
        "prcp":  [0.0]  * n,
        "snow":  [0.0]  * n,
    })


# ── Feature engineering ───────────────────────────────────────────────────────

def test_build_features_produces_required_columns():
    df = build_features(make_raw_df())
    for col in ["travel_time_sec", "stop_pair", "hour", "minute", "day_of_week",
                "month", "is_weekend", "is_peak_am", "is_peak_pm",
                "is_precipitating", "is_snowing", "heavy_snow"]:
        assert col in df.columns, f"Missing column: {col}"


def test_build_features_travel_time_is_positive():
    df = build_features(make_raw_df())
    assert (df["travel_time_sec"] > 0).all()


def test_build_features_drops_zero_and_negative_travel_times():
    raw = make_raw_df(5)
    # Inject a row where arrival == departure (0s travel time)
    raw.loc[0, "to_stop_arrival_datetime"] = raw.loc[0, "from_stop_departure_datetime"]
    df = build_features(raw)
    assert (df["travel_time_sec"] > 0).all()


def test_build_features_caps_dwell_at_600():
    raw = make_raw_df(5)
    raw["dwell_time_sec"] = [700.0, -10.0, 30.0, 600.0, 0.0]
    df = build_features(raw)
    assert df["dwell_time_sec"].max() <= 600
    assert df["dwell_time_sec"].min() >= 0


def test_peak_flags_are_correct():
    raw = make_raw_df(1)
    # Set departure to 8 AM — should be AM peak
    raw["from_stop_departure_datetime"] = [pd.Timestamp("2024-10-15 08:00:00", tz="UTC")]
    raw["to_stop_arrival_datetime"]     = [pd.Timestamp("2024-10-15 08:01:30", tz="UTC")]
    df = build_features(raw)
    assert df["is_peak_am"].iloc[0] == 1
    assert df["is_peak_pm"].iloc[0] == 0


# ── Dwell outlier filtering (dataset_creation logic) ─────────────────────────

def test_dwell_outlier_filter_removes_negatives_and_long_waits():
    df = pd.DataFrame({
        "dwell_time_sec": [-5.0, 0.0, 30.0, 599.0, 600.0, 601.0, np.nan]
    })
    filtered = df[
        df["dwell_time_sec"].isna() |
        ((df["dwell_time_sec"] >= 0) & (df["dwell_time_sec"] < 600))
    ]
    assert set(filtered["dwell_time_sec"].dropna().tolist()) == {0.0, 30.0, 599.0}
    assert filtered["dwell_time_sec"].isna().sum() == 1  # NaN is kept


# ── BU calendar logic ─────────────────────────────────────────────────────────

def test_class_day_during_semester():
    df = pd.DataFrame({
        "from_stop_departure_datetime": [pd.Timestamp("2024-10-15 10:00:00")]  # Fall 2024, Tuesday
    })
    result = add_bu_semester_logic(df)
    assert result["is_bu_class_day"].iloc[0] == 1


def test_no_class_day_on_weekend():
    df = pd.DataFrame({
        "from_stop_departure_datetime": [pd.Timestamp("2024-10-19 10:00:00")]  # Saturday
    })
    result = add_bu_semester_logic(df)
    assert result["is_bu_class_day"].iloc[0] == 0


def test_no_class_day_on_holiday():
    df = pd.DataFrame({
        "from_stop_departure_datetime": [pd.Timestamp("2024-11-27 10:00:00")]  # Thanksgiving
    })
    result = add_bu_semester_logic(df)
    assert result["is_bu_class_day"].iloc[0] == 0


def test_no_class_day_outside_semester():
    df = pd.DataFrame({
        "from_stop_departure_datetime": [pd.Timestamp("2024-07-04 10:00:00")]  # Summer, no semester
    })
    result = add_bu_semester_logic(df)
    assert result["is_bu_class_day"].iloc[0] == 0


# ── Surge flag logic ──────────────────────────────────────────────────────────

def test_surge_flag_mwf_class_end():
    # MWF 11:00 end — should be surge
    t = pd.Timestamp("2024-10-16 11:05:00")  # Wednesday
    assert get_surge_flag(t) == 1


def test_surge_flag_tr_triple_wave():
    # Tuesday 10:45 — triple-wave hotspot, should return 2
    t = pd.Timestamp("2024-10-15 10:50:00")  # Tuesday
    assert get_surge_flag(t) == 2


def test_no_surge_outside_class_times():
    # Wednesday at 3 AM — no surge
    t = pd.Timestamp("2024-10-16 03:00:00")
    assert get_surge_flag(t) == 0
