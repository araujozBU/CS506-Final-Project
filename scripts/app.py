import streamlit as st
import pandas as pd
import pickle
import json
import os
import altair as alt

from model import predict_travel_time, load_training_data_from_huggingface, build_features

st.set_page_config(
    page_title="BU Green Line Planner",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.mbta-header {
    background: linear-gradient(135deg, #00843D 0%, #005B2A 100%);
    color: white; padding: 22px 32px 18px; border-radius: 14px;
    margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,132,61,0.2);
}
.mbta-header h1 { margin: 0 0 4px; font-size: 1.8rem; font-weight: 800; }
.mbta-header p  { margin: 0; opacity: 0.85; font-size: 0.88rem; }

div[data-testid="metric-container"] {
    border-left: 4px solid #00843D;
    border-radius: 8px;
    padding: 12px 16px !important;
    background: white !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
UNIQUE_STATIONS = [
    "Kenmore", "Blandford St", "BU East",
    "BU Central", "Amory", "Babcock"
]

ROUTE_STATIONS = {
    "Westbound": [
        "Hynes", "Kenmore", "Blandford St", "BU East",
        "BU Central", "Amory", "Babcock",
    ],
    "Eastbound": [
        "Packard's", "Babcock", "Amory", "BU Central",
        "BU East", "Blandford St", "Kenmore",
    ],
}

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

DAY_NAMES   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_is_student_surge(hour: int, minute: int, day_of_week: int) -> int:
    """
    Mirrors dataset_creation.get_surge_flag() exactly.
    Returns 1 if the given time falls in a BU class-end surge window, else 0.
    Surge=2 (hotspot) in the original is also treated as active (>=1).
    """
    if day_of_week >= 5:
        return 0
    if day_of_week in [0, 2, 4]:  # MWF
        if (hour == 8  and 50 <= minute <= 59) or (hour == 9  and  0 <= minute <= 10): return 1
        if (hour == 9  and 55 <= minute <= 59) or (hour == 10 and  0 <= minute <= 15): return 1
        if  hour == 11 and  0 <= minute <= 15: return 1
        if  hour == 12 and 10 <= minute <= 25: return 1
        if  hour == 13 and 25 <= minute <= 40: return 1
        if  hour == 14 and 15 <= minute <= 30: return 1
        if  hour == 15 and 20 <= minute <= 35: return 1
        if  hour == 16 and 25 <= minute <= 40: return 1
    elif day_of_week in [1, 3]:  # TR
        if  hour == 9  and 15 <= minute <= 30: return 1
        if (hour == 10 and 45 <= minute <= 59) or (hour == 11 and  0 <= minute <= 10): return 1
        if  hour == 12 and 15 <= minute <= 30: return 1
        if (hour == 13 and 45 <= minute <= 59) or (hour == 14 and  0 <= minute <= 10): return 1
        if  hour == 15 and 15 <= minute <= 30: return 1
        if (hour == 16 and 45 <= minute <= 59) or (hour == 17 and  0 <= minute <= 10): return 1
        if  hour == 18 and 15 <= minute <= 30: return 1
    return 0


def format_time(seconds):
    if seconds is None or pd.isna(seconds):
        return "N/A"
    seconds = int(seconds)
    return f"{seconds // 60}m {seconds % 60}s"


def segment_traffic_html(segments_data):
    """Color-coded route map: green = on time, yellow = slight, red = heavy delay."""
    if not segments_data:
        return (
            '<div style="background:white;border-radius:12px;padding:18px 22px;'
            'box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:4px;'
            'color:#546e7a;font-size:0.9rem;">'
            'No route segments available for the selected stations.'
            '</div>'
        )

    def seg_color(ratio):
        if ratio <= -0.05: return "#2E7D32"
        if ratio <=  0.05: return "#66BB6A"
        if ratio <=  0.15: return "#FFC107"
        if ratio <=  0.30: return "#FF7043"
        return "#E53935"

    stops = [s["Segment"].split(" → ")[0] for s in segments_data]
    stops.append(segments_data[-1]["Segment"].split(" → ")[1])

    ratios = [
        (s["travel_sec"] - s["base_sec"]) / s["base_sec"]
        if s["base_sec"] is not None and s["base_sec"] > 0 else 0
        for s in segments_data
    ]

    W = "56px"  # fixed width per stop cell

    # Row 1 — % delay labels above each segment
    r1 = []
    for i in range(len(stops)):
        r1.append(f'<div style="min-width:{W};flex-shrink:0;"></div>')
        if i < len(ratios):
            c   = seg_color(ratios[i])
            lbl = f"+{ratios[i]*100:.0f}%" if ratios[i] > 0 else f"{ratios[i]*100:.0f}%"
            r1.append(
                f'<div style="flex:1;text-align:center;font-size:0.68rem;'
                f'font-weight:700;color:{c};">{lbl}</div>'
            )

    # Row 2 — circles + colored bars
    r2 = []
    for i, _ in enumerate(stops):
        r2.append(
            f'<div style="min-width:{W};flex-shrink:0;display:flex;'
            f'justify-content:center;align-items:center;">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:#37474f;border:2.5px solid white;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.25);"></div></div>'
        )
        if i < len(ratios):
            c = seg_color(ratios[i])
            r2.append(
                f'<div style="flex:1;height:11px;background:{c};'
                f'border-radius:3px;box-shadow:inset 0 1px 2px rgba(0,0,0,0.1);"></div>'
            )

    # Row 3 — stop name labels
    r3 = []
    for i, stop in enumerate(stops):
        r3.append(
            f'<div style="min-width:{W};flex-shrink:0;text-align:center;'
            f'font-size:0.63rem;color:#546e7a;padding-top:4px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{stop}</div>'
        )
        if i < len(ratios):
            r3.append(f'<div style="flex:1;"></div>')

    legend_items = [
        ("#2E7D32", "Faster"), ("#66BB6A", "On time"),
        ("#FFC107", "+5–15%"), ("#FF7043", "+15–30%"), ("#E53935", ">30%"),
    ]
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:5px;margin-right:14px;">'
        f'<div style="width:20px;height:7px;border-radius:3px;background:{c};"></div>'
        f'<span style="font-size:0.65rem;color:#666;">{l}</span></div>'
        for c, l in legend_items
    )

    row = lambda parts, extra="": (
        f'<div style="display:flex;align-items:center;{extra}">'
        + "".join(parts) + '</div>'
    )

    return (
        '<div style="background:white;border-radius:12px;padding:18px 22px 14px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:4px;">'
        '<p style="font-size:0.68rem;font-weight:700;letter-spacing:1.3px;'
        'text-transform:uppercase;color:#37474f;margin:0 0 10px;">Route Conditions</p>'
        + row(r1, "margin-bottom:4px;")
        + row(r2, "margin-bottom:2px;")
        + row(r3)
        + '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:12px;'
        'padding-top:10px;border-top:1px solid #eceff1;">'
        + legend + '</div></div>'
    )


def stop_indicator_html(idx, from_stop, to_stop):
    """Vertical line segment + circle for one stop row."""
    fi = UNIQUE_STATIONS.index(from_stop) if from_stop in UNIQUE_STATIONS else -1
    ti = UNIQUE_STATIONS.index(to_stop)   if to_stop   in UNIQUE_STATIONS else -1
    lo = min(fi, ti) if fi >= 0 and ti >= 0 else -1
    hi = max(fi, ti) if fi >= 0 and ti >= 0 else -1

    stop     = UNIQUE_STATIONS[idx]
    is_from  = stop == from_stop
    is_to    = stop == to_stop
    on_route = lo >= 0 and lo <= idx <= hi
    is_first = idx == 0
    is_last  = idx == len(UNIQUE_STATIONS) - 1

    above_color = "#00843D" if (lo >= 0 and lo < idx <= hi) else "#cfd8dc"
    below_color = "#00843D" if (lo >= 0 and lo <= idx < hi) else "#cfd8dc"

    if is_from:
        dot_color, dot_sz = "#005B2A", "18px"
    elif is_to:
        dot_color, dot_sz = "#E87722", "18px"
    elif on_route:
        dot_color, dot_sz = "#00843D", "13px"
    else:
        dot_color, dot_sz = "#90a4ae", "10px"

    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;'
        f'height:100%;min-height:44px;padding:0 8px;">'
        f'<div style="flex:1;width:4px;border-radius:2px;'
        f'background:{"transparent" if is_first else above_color};"></div>'
        f'<div style="width:{dot_sz};height:{dot_sz};border-radius:50%;'
        f'background:{dot_color};border:2px solid white;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.2);flex-shrink:0;"></div>'
        f'<div style="flex:1;width:4px;border-radius:2px;'
        f'background:{"transparent" if is_last else below_color};"></div>'
        f'</div>'
    )


# ── Data loading ──────────────────────────────────────────────────────────────
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

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("sel_from", "Kenmore"), ("sel_to", "Babcock"), ("select_mode", "from")]:
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state["sel_from"] not in UNIQUE_STATIONS:
    st.session_state["sel_from"] = "Kenmore"
if st.session_state["sel_to"] not in UNIQUE_STATIONS:
    st.session_state["sel_to"] = "Babcock"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="mbta-header">
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="width:58px;height:58px;border-radius:50%;background:white;
         display:flex;align-items:center;justify-content:center;flex-shrink:0;
         box-shadow:0 2px 10px rgba(0,0,0,0.25);">
      <svg width="34" height="34" viewBox="0 0 34 34" xmlns="http://www.w3.org/2000/svg">
        <rect x="1"  y="2"  width="32" height="9"  rx="2.5" fill="#00843D"/>
        <rect x="11" y="2"  width="12" height="30" rx="2.5" fill="#00843D"/>
      </svg>
    </div>
    <div>
      <h1 style="margin:0 0 4px;font-size:1.85rem;font-weight:800;letter-spacing:0.3px;">
        BU Green Line Planner</h1>
      <p style="margin:0;opacity:0.85;font-size:0.88rem;">
        MBTA Green Line B-Branch &nbsp;&middot;&nbsp;
        Travel time predictions powered by weather &amp; BU academic calendar
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.65], gap="large")

with left:
    st.markdown("**Route**")
    rc1, rc2 = st.columns(2)
    with rc1:
        sel_from = st.selectbox(
            "From Stop", UNIQUE_STATIONS,
            index=UNIQUE_STATIONS.index(st.session_state["sel_from"]),
        )
        st.session_state["sel_from"] = sel_from
    with rc2:
        sel_to = st.selectbox(
            "To Stop", UNIQUE_STATIONS,
            index=UNIQUE_STATIONS.index(st.session_state["sel_to"]),
        )
        st.session_state["sel_to"] = sel_to

    st.markdown("**Time**")
    hc1, hc2 = st.columns(2)
    with hc1:
        hour = st.slider("Hour", 0, 23, 8, format="%d:00")
    with hc2:
        minute = st.slider("Minute", 0, 59, 0, step=1, format="%d min")
    tc1, tc2 = st.columns(2)
    with tc1:
        day_of_week = st.selectbox(
            "Day of Week", list(range(7)),
            format_func=lambda x: DAY_NAMES[x]
        )
    with tc2:
        month = st.selectbox(
            "Month", list(range(1, 13)),
            format_func=lambda x: MONTH_NAMES[x - 1],
            index=8
        )

    st.markdown("**Weather**")
    temp = st.slider("Temperature (°C)", -15.0, 35.0, 15.0, step=0.5)
    wc1, wc2 = st.columns(2)
    with wc1:
        prcp = st.slider("Precipitation (mm)", 0.0, 20.0, 0.0, step=0.5)
    with wc2:
        snow = st.slider("Snow (mm)", 0.0, 30.0, 0.0, step=0.5)

    st.markdown("**BU Context**")
    is_bu_class_day      = st.checkbox("BU Class Day", value=True)
    is_active_class_time = int(is_bu_class_day and (8 <= hour < 18))
    is_student_surge     = int(is_bu_class_day and bool(compute_is_student_surge(hour, minute, day_of_week)))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Travel Time", use_container_width=True, type="primary")

with right:
    from_stop = st.session_state["sel_from"]
    to_stop   = st.session_state["sel_to"]
    mode      = st.session_state["select_mode"]

    # Mode toggle + label
    _, mc, _ = st.columns([1, 4, 1])
    with mc:
        mode_text = "Click a stop to set departure" if mode == "from" else "Click a stop to set arrival"
        st.markdown(
            f'<p style="text-align:center;font-size:0.8rem;color:#555;margin-bottom:6px;">'
            f'{mode_text}</p>',
            unsafe_allow_html=True,
        )
        t1, t2 = st.columns(2)
        with t1:
            if st.button("Set departure", use_container_width=True):
                st.session_state["select_mode"] = "from"
                st.rerun()
        with t2:
            if st.button("Set arrival", use_container_width=True):
                st.session_state["select_mode"] = "to"
                st.rerun()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Vertical stop diagram — one row per stop
        for i, stop in enumerate(UNIQUE_STATIONS):
            ic, bc = st.columns([1, 6])
            is_from = stop == from_stop
            is_to   = stop == to_stop
            with ic:
                st.markdown(stop_indicator_html(i, from_stop, to_stop), unsafe_allow_html=True)
            with bc:
                if is_from:
                    label = f"{stop}  —  departure"
                elif is_to:
                    label = f"{stop}  —  arrival"
                else:
                    label = stop
                if st.button(label, key=f"vbtn_{stop}", use_container_width=True):
                    if st.session_state["select_mode"] == "from":
                        st.session_state["sel_from"] = stop
                        st.session_state["select_mode"] = "to"
                    else:
                        st.session_state["sel_to"] = stop
                        st.session_state["select_mode"] = "from"
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    ci1, ci2, ci3, ci4 = st.columns(4)
    weather_str = "Rain" if prcp > 0 else ("Snow" if snow > 0 else "Clear")
    with ci1:
        st.metric("Temp", f"{temp:.0f}°C")
    with ci2:
        st.metric("Weather", weather_str)
    with ci3:
        st.metric("Day", DAY_NAMES[day_of_week])
    with ci4:
        st.metric("Hour", f"{hour:02d}:00")


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    from_stop_name = st.session_state["sel_from"]
    to_stop_name   = st.session_state["sel_to"]

    from_idx = UNIQUE_STATIONS.index(from_stop_name)
    to_idx   = UNIQUE_STATIONS.index(to_stop_name)

    if from_idx == to_idx:
        st.error("Please select different start and end stops.")
        st.stop()

    direction = "Westbound" if to_idx > from_idx else "Eastbound"

    if from_stop_name not in PLATFORM_IDS[direction] or to_stop_name not in PLATFORM_IDS[direction]:
        st.error(
            f"No {direction} data for {from_stop_name} to {to_stop_name}. "
            "Try a different stop pair."
        )
        st.stop()

    route_stations = ROUTE_STATIONS[direction]
    try:
        route_start_idx = route_stations.index(from_stop_name)
        route_end_idx = route_stations.index(to_stop_name)
    except ValueError:
        st.error("Could not build a valid route for the selected stop pair.")
        st.stop()

    if route_start_idx == route_end_idx:
        st.error("Please select different start and end stops.")
        st.stop()

    step = 1 if route_start_idx < route_end_idx else -1
    route_stop_names = route_stations[route_start_idx:route_end_idx + step:step]

    if len(route_stop_names) < 2:
        st.error("Could not build a valid route for the selected stop pair.")
        st.stop()

    is_weekend       = int(day_of_week >= 5)
    is_peak_am       = int(7 <= hour <= 9)
    is_peak_pm       = int(16 <= hour <= 19)
    df_ref_local     = load_reference_data()

    total_travel  = 0.0
    total_dwell   = 0.0
    segments_data = []

    for i in range(len(route_stop_names) - 1):
        seg_from = route_stop_names[i]
        seg_to   = route_stop_names[i + 1]
        from_id  = PLATFORM_IDS[direction][seg_from]
        to_id    = PLATFORM_IDS[direction][seg_to]

        pred = predict_travel_time(
            model_dwell, model_travel, le, df_ref_local,
            from_id, to_id,
            hour=hour, day_of_week=day_of_week,
            temp=temp, prcp=prcp, snow=snow,
            is_bu_class_day=int(is_bu_class_day),
            is_active_class_time=int(is_active_class_time),
            is_student_surge=int(is_student_surge),
            month=month,
        )

        travel_sec = pred["predicted_sec"]
        dwell_sec  = pred["predicted_dwell_sec"]
        base_sec   = pred["baseline_sec"]

        total_travel += travel_sec
        if i > 0:
            total_dwell += dwell_sec

        segments_data.append({
            "Segment":    f"{seg_from} → {seg_to}",
            "travel_sec": travel_sec,
            "dwell_sec":  dwell_sec,
            "base_sec":   base_sec,
        })
        row = pred["feature_row"]

    pred_sec = total_travel + total_dwell

    # ── Hour-of-day curve (all 24 hours, same route + conditions) ─────────
    hourly_rows = []
    for h in range(24):
        h_travel, h_dwell = 0.0, 0.0
        h_active  = int(is_bu_class_day and (8 <= h <= 18))
        h_surge   = int(is_bu_class_day and bool(compute_is_student_surge(h, 30, day_of_week)))
        h_peak_am = int(7 <= h <= 9)
        h_peak_pm = int(16 <= h <= 19)
        for j in range(len(route_stop_names) - 1):
            sf = route_stop_names[j]
            st_ = route_stop_names[j + 1]
            p = predict_travel_time(
                model_dwell, model_travel, le, df_ref_local,
                PLATFORM_IDS[direction][sf], PLATFORM_IDS[direction][st_],
                hour=h, day_of_week=day_of_week,
                temp=temp, prcp=prcp, snow=snow,
                is_bu_class_day=int(is_bu_class_day),
                is_active_class_time=h_active,
                is_student_surge=h_surge,
                month=month,
            )
            h_travel += p["predicted_sec"]
            if j > 0:
                h_dwell += p["predicted_dwell_sec"]
        hourly_rows.append({
            "Hour": h,
            "TotalSec": h_travel + h_dwell,
            "Label": f"{h:02d}:00",
        })
    df_hourly = pd.DataFrame(hourly_rows)
    baseline_sec = df_hourly["TotalSec"].min()
    df_hourly["DeltaSec"] = (df_hourly["TotalSec"] - baseline_sec).round(1)

    # ── Metrics ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Results")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            f"Total — {from_stop_name} to {to_stop_name} ({direction})",
            format_time(pred_sec)
        )
    with m2:
        st.metric("Travel Time", format_time(total_travel))
    with m3:
        st.metric(
            "Platform Wait", format_time(total_dwell),
            delta=f"{int(total_dwell)}s dwell at intermediate stops",
            delta_color="off",
        )

    # ── Route conditions map ──────────────────────────────────────────────
    st.markdown(segment_traffic_html(segments_data), unsafe_allow_html=True)

    # ── Hour-of-day chart ─────────────────────────────────────────────────
    st.markdown("### Extra Delay by Hour of Day")
    st.caption(
        f"Seconds added above the fastest possible trip — {from_stop_name} to {to_stop_name}, "
        f"same weather & day conditions. Zero = best-case trip time."
    )

    line = (
        alt.Chart(df_hourly)
        .mark_area(
            line={"color": "#00843D", "strokeWidth": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(0,132,61,0.3)", offset=0),
                    alt.GradientStop(color="rgba(0,132,61,0.02)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("Hour:Q", title="Hour of Day",
                    axis=alt.Axis(values=list(range(0, 24, 2)), format="02d")),
            y=alt.Y("DeltaSec:Q", title="Extra seconds vs fastest hour",
                    scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("Label:N", title="Time"),
                alt.Tooltip("DeltaSec:Q", format=".0f", title="Extra seconds"),
            ],
        )
        .properties(height=220)
    )

    rule = (
        alt.Chart(pd.DataFrame({"Hour": [hour]}))
        .mark_rule(color="#E87722", strokeWidth=2, strokeDash=[4, 3])
        .encode(x="Hour:Q")
    )

    dot = (
        alt.Chart(df_hourly[df_hourly["Hour"] == hour])
        .mark_point(color="#E87722", size=80, filled=True)
        .encode(
            x="Hour:Q",
            y="DeltaSec:Q",
            tooltip=[
                alt.Tooltip("Label:N", title="Selected"),
                alt.Tooltip("DeltaSec:Q", format=".0f", title="Extra seconds"),
            ],
        )
    )

    st.altair_chart(line + rule + dot, use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────
    df_seg = pd.DataFrame(segments_data)

    st.markdown("### Segment Breakdown")
    ch_left, ch_right = st.columns([3, 2])

    with ch_left:
        df_chart = df_seg.copy()
        df_chart["base_sec"] = df_chart["base_sec"].fillna(df_chart["travel_sec"])
        df_melt = (
            df_chart[["Segment", "travel_sec", "base_sec"]]
            .rename(columns={"travel_sec": "Predicted", "base_sec": "Baseline"})
            .melt("Segment", var_name="Type", value_name="Seconds")
        )
        bar = (
            alt.Chart(df_melt)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Segment:N", title="",
                        axis=alt.Axis(labelAngle=-20, labelLimit=130)),
                y=alt.Y("Seconds:Q", title="Seconds"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Predicted", "Baseline"],
                        range=["#00843D", "#b0bec5"]
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                xOffset="Type:N",
                tooltip=[
                    "Segment:N", "Type:N",
                    alt.Tooltip("Seconds:Q", format=".0f")
                ],
            )
            .properties(height=280, title="Predicted vs Baseline Travel Time per Segment")
        )
        st.altair_chart(bar, use_container_width=True)

    with ch_right:
        if total_dwell > 0:
            df_donut = pd.DataFrame({
                "Category": ["Travel Time", "Platform Wait"],
                "Seconds":  [total_travel, total_dwell],
            })
            donut = (
                alt.Chart(df_donut)
                .mark_arc(innerRadius=55, outerRadius=100)
                .encode(
                    theta=alt.Theta("Seconds:Q"),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            domain=["Travel Time", "Platform Wait"],
                            range=["#00843D", "#E87722"],
                        ),
                        legend=alt.Legend(orient="bottom"),
                    ),
                    tooltip=["Category:N", alt.Tooltip("Seconds:Q", format=".0f")],
                )
                .properties(height=260, title="Time Breakdown")
            )
            st.altair_chart(donut, use_container_width=True)
        else:
            st.info("Single segment — all time is travel time.")

    # ── Table ─────────────────────────────────────────────────────────────
    st.markdown("### Detailed Table")
    df_display = pd.DataFrame([{
        "Segment":          d["Segment"],
        "Predicted Travel": format_time(d["travel_sec"]),
        "Baseline Travel":  format_time(d["base_sec"]),
        "Predicted Dwell":  format_time(d["dwell_sec"]),
    } for d in segments_data])
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ── Conditions + debug ────────────────────────────────────────────────
    cc1, cc2 = st.columns(2)
    with cc1:
        st.info(
            "**Trip Conditions**\n\n"
            f"- Direction: {direction}\n"
            f"- Weather: {'Rain' if prcp > 0 else 'Snow' if snow > 0 else 'Clear'}\n"
            f"- BU Student Surge: {'Active' if is_student_surge else 'Inactive'}\n"
            f"- Period: {'AM Peak' if is_peak_am else 'PM Peak' if is_peak_pm else 'Off-Peak'}"
        )
    with cc2:
        with st.expander("Show model feature inputs"):
            st.json({
                "hour": hour, "minute": minute, "day_of_week": day_of_week,
                "month": month, "is_weekend": is_weekend,
                "is_peak_am": is_peak_am, "is_peak_pm": is_peak_pm,
                "temp": temp, "prcp": prcp, "snow": snow,
                "is_bu_class_day": int(is_bu_class_day),
                "is_active_class_time": int(is_active_class_time),
                "is_student_surge": int(is_student_surge),
            })
