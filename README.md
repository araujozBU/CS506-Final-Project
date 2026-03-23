# The BU Commuter’s Guide to the B-Branch: Predicting Transit Reliability
### *A Segment-Level Analysis of Weather and Academic Schedule Impacts*

## 1. Project Description & Motivation
While the MBTA Green Line is the lifeblood of the Boston University campus, its reliability is notoriously volatile. Because the B-Branch operates at street level along Commonwealth Avenue, it is uniquely susceptible to two types of "shocks": 
1. **Meteorological Shocks:** Rain and snow affecting track friction, visibility, and "portal" entry speeds.
2. **Social Shocks:** Massive "Student Surges" during BU class transition windows.

This project aims to move beyond generic delay alerts by building a context-aware predictive model. We treat a trip not as a single event, but as a sequence of **Running Time** (inter-station movement) and **Dwell Time** (platform boarding) segments. By modeling these separately, we provide a granular "Trip Calculator" that accounts for exactly how weather and the BU Academic Calendar impact a student's specific commute.

---

## 2. Project Goals
* **Segment-Level Accuracy:** Predict Running and Dwell times for key BU segments (Babcock St to Kenmore) with a Mean Absolute Error (MAE) of less than 45 seconds.
* **Component Decomposition:** Quantify the delay caused by "Dwell Time" (boarding volume) vs. "Running Time" (train speed).
* **Contextual Forecasting:** Successfully identify "Student Surge" patterns by correlating delays with the BU class schedule and semester calendar.
* **Trip Summation:** Create a tool that calculates the total predicted commute:  
    $$Total\ Time = Dwell_{start} + \sum(Running_{segments}) + \sum(Dwell_{intermediate})$$

---

## 3. Data Collection Plan

### Data Sources
* **MBTA Blue Book / Open Data Portal:** We are utilizing the [MBTA Rapid Transit Travel Times dataset](https://mbta-massdot.opendata.arcgis.com/datasets/5f71a5c035fc4a4dad1b7fa73ba27ef8/about) to extract high-resolution event data (Arrivals vs. Departures) for the B-Branch.
* **Meteostat / OpenWeather API:** Historical hourly weather data for Boston (Precipitation, Snow Depth, Temperature, and Wind Speed).
* **BU Academic Context:** A manually curated dataset based on the **BU 2025-2026 Academic Calendar**, including holiday flags, Spring Break dates, and the standard 10-minute class transition windows.

### Feature Engineering
We will transform raw timestamps into a "Context Matrix" including:
* **`is_class_change`**: Binary flag for the 10-minute windows around BU class starts/ends.
* **`is_semester`**: Binary flag to exclude summer/winter breaks.
* **`direction_id`**: To account for directional flow (Inbound/Eastbound in the morning vs. Outbound/Westbound in the afternoon).

---

## 4. Modeling & Implementation
We will implement a **Random Forest Regressor**. This ensemble method is ideal for handling the non-linear relationships in transit data—such as the "tipping point" where light snow causes disproportionately large delays at the Blandford St portal entry.

The model will be a **Dual-Target Regressor**, predicting:
1.  **Running Time:** Primarily influenced by weather and track conditions.
2.  **Dwell Time:** Primarily influenced by class schedules and station location (e.g., BU Central vs. Blandford).

---

## 5. Visualization Plan
* **Stacked Component Analysis:** A visualization showing a "Trip Breakdown" where users can see time spent moving vs. time spent at platforms.
* **The "Student Surge" Heatmap:** A temporal map showing how dwell times at BU-specific stations spike in sync with the university's bell schedule.
* **Weather Sensitivity Index:** A comparison of how different segments react to precipitation.

---

## 6. Test Plan
We will use a **Temporal Train-Test Split** to ensure the model's real-world utility:
* **Training Set:** Data from the first half of the semester (e.g., September – October).
* **Testing Set:** Data from the latter half (November – December). 
This allows us to evaluate how the model handles the transition from "Fall Rain" to "Winter Snow" and changes in ridership during "Finals Week."

---

## Team & Repository
* **Group Size:** 5 students  
* **GitHub Repository:** [https://github.com/araujozBU/CS506-Final-Project](https://github.com/araujozBU/CS506-Final-Project)

### Team Members
* **Zaki Araujo** (araujoz@bu.edu)
* **Adrian Dybacki** (adybacki@bu.edu)
* **Andrew Botolino** (botolino@bu.edu)
* **Kuba Rozwadowski** (kubaroz@bu.edu)
* **Rohan Chablani** (rohan204@bu.edu)