import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

from model import predict_travel_time, load_training_data_from_huggingface, build_features

def format_time(seconds):
    if seconds is None or pd.isna(seconds):
        return "N/A"
    seconds = int(seconds)
    m = seconds // 60
    s = seconds % 60
    return f"{m}m {s}s"

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "model_artifacts")
    
    with open(os.path.join(artifacts_dir, "model_dwell.pkl"), "rb") as f:
        model_dwell = pickle.load(f)
    with open(os.path.join(artifacts_dir, "model_travel.pkl"), "rb") as f:
        model_travel = pickle.load(f)
    with open(os.path.join(artifacts_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(artifacts_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
        
    return model_dwell, model_travel, le, metadata


@st.cache_data
def load_reference_data():
    df_ref, _ = load_training_data_from_huggingface()
    df_ref = build_features(df_ref)
    return df_ref


model_dwell, model_travel, le, metadata = load_artifacts()
df_ref = load_reference_data()

stop_names = metadata["stop_names"]
stop_sequence = metadata["stop_sequence"]

# --- 2. UI Layout ---
st.title("🚇 BU Green Line Trip Calculator")
st.markdown("Predict travel times for the B-Branch based on weather and BU Academic Calendar constraints.")

st.sidebar.header("Commute Settings")

# Stop selection
st.sidebar.subheader("Route")

UNIQUE_STATIONS = [
    "Hynes", "Kenmore", "Blandford St", "BU East", 
    "BU Central", "Amory", "Babcock", "Packard's"
]

PLATFORM_IDS = {
    "Westbound": {
        "Hynes": "70153", "Kenmore": "71151", "Blandford St": "70149",
        "BU East": "70147", "BU Central": "70145", "Amory": "170141", "Babcock": "170137",
    },
    "Eastbound": {
        "Packard's": "70134", "Babcock": "170136", "Amory": "170140",
        "BU Central": "70144", "BU East": "70146", "Blandford St": "70148", "Kenmore": "71150"
    }
}

from_stop_name = st.sidebar.selectbox("From Stop", UNIQUE_STATIONS, index=1)
to_stop_name = st.sidebar.selectbox("To Stop", UNIQUE_STATIONS, index=6)

# Time selection
st.sidebar.subheader("Time Context")
hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 8)
day_of_week = st.sidebar.selectbox(
    "Day of Week", 
    list(range(7)), 
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
)
month = st.sidebar.slider("Month (1-12)", 1, 12, 9)

# Weather selection
st.sidebar.subheader("Weather Conditions")
temp = st.sidebar.slider("Temperature (°C)", -15.0, 35.0, 15.0)
prcp = st.sidebar.slider("Precipitation (mm)", 0.0, 20.0, 0.0)
snow = st.sidebar.slider("Snow (mm)", 0.0, 30.0, 0.0)

# BU Context
st.sidebar.subheader("BU Campus Context")
is_bu_class_day = st.sidebar.checkbox("BU Class Day", value=True)

# Infer class period flags if it's a BU class day (e.g. 8am-6pm is active, specific hours are surges)
is_active_class_time = is_bu_class_day and (8 <= hour <= 18)
is_student_surge = is_bu_class_day and (hour in [10, 12, 14, 16])

# --- 3. Inference Logic ---
if st.button("Predict Travel Time"):
    from_idx = UNIQUE_STATIONS.index(from_stop_name)
    to_idx = UNIQUE_STATIONS.index(to_stop_name)
    
    if from_idx == to_idx:
        st.error("Please select different start and end stops.")
        st.stop()
        
    direction = "Westbound" if to_idx > from_idx else "Eastbound"
    
    if from_stop_name not in PLATFORM_IDS[direction] or to_stop_name not in PLATFORM_IDS[direction]:
        st.error(f"⚠️ The model does not have {direction} data spanning this specific combination ({from_stop_name} to {to_stop_name}). Please try another pair.")
        st.stop()
        
    # Determine segment sequence 
    step = 1 if direction == "Westbound" else -1
    route_stop_names = UNIQUE_STATIONS[from_idx : to_idx + step : step]
        
    # Derived features based on user inputs
    minute = 0 
    is_weekend = int(day_of_week >= 5)
    is_peak_am = int(7 <= hour <= 9)
    is_peak_pm = int(16 <= hour <= 19)
    
    df_ref = load_reference_data()

    total_travel_time_sec = 0.0
    total_dwell_time_sec = 0.0
    segments_str = []

    for i in range(len(route_stop_names) - 1):
        seg_from = route_stop_names[i]
        seg_to = route_stop_names[i+1]
        
        from_id = PLATFORM_IDS[direction][seg_from]
        to_id = PLATFORM_IDS[direction][seg_to]

        pred_result = predict_travel_time(
            model_dwell, model_travel, le, df_ref,
            from_id, to_id,
            hour=hour,
            day_of_week=day_of_week,
            temp=temp,
            prcp=prcp,
            snow=snow,
            is_bu_class_day=int(is_bu_class_day),
            is_active_class_time=int(is_active_class_time),
            is_student_surge=int(is_student_surge),
            month=month,
        )

        dwell_time_sec = pred_result["predicted_dwell_sec"]
        segment_sec = pred_result["predicted_sec"]
        total_travel_time_sec += segment_sec
        if i > 0:
            total_dwell_time_sec += dwell_time_sec

        segments_str.append({
            "Segment": f"{seg_from} ➔ {seg_to}",
            "Predicted Dwell": format_time(dwell_time_sec),
            "Predicted Travel Time": format_time(segment_sec),
            "Baseline Travel Time": format_time(pred_result["baseline_sec"])
        })
        row = pred_result["feature_row"] # Keep the last row for display
        
    pred_sec = total_travel_time_sec

    pred_sec = total_travel_time_sec + total_dwell_time_sec

    # --- 4. Display Results ---
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label=f"Total Commute: {from_stop_name} ➔ {to_stop_name} ({direction})",
            value=format_time(pred_sec),
            delta=f"Includes {int(total_dwell_time_sec)}s of platform waiting",
            delta_color="off"
        )
    with col2:
        st.info("💡 **Insights:**\n" +
                f"- Weather constraint active: {'Yes' if prcp > 0 or snow > 0 else 'No'}\n" +
                f"- BU Student Surge factor: {'Active' if is_student_surge else 'Inactive'}")

    st.subheader("🔍 Trip Segment Breakdown")
    st.table(pd.DataFrame(segments_str))

    with st.expander("Show last model feature inputs"):
        st.write({
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_peak_am": is_peak_am,
            "is_peak_pm": is_peak_pm,
            "temp": temp,
            "prcp": prcp,
            "snow": snow,
            "is_bu_class_day": int(is_bu_class_day),
            "is_active_class_time": int(is_active_class_time),
            "is_student_surge": int(is_student_surge),
        })
