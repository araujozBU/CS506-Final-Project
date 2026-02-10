# Green Line Delay Prediction Using Weather and MBTA Data

## Project Description & Motivation

Weather conditions impact the reliability of public transit, especially on the MBTA Green Line. Riders frequently experience delays due to rain, snow, extreme temperatures, and other environmental factors. However, these delays are often communicated only after they occur.

The goal of this project is to build a system that combines **historical MBTA Green Line performance data** with **weather data from the OpenWeather API** to predict the likelihood of service delays. By identifying patterns between weather conditions and transit performance, this project aims to provide insight into when delays are more likely to occur.



## Project Goals

1. Predict whether a Green Line trip will experience a delay based on weather conditions.
2. Quantify the relationship between weather features (precipitation, temperature, wind speed, snowfall) and transit delays.
3. Evaluate model performance using metrics such as accuracy, precision, recall, and F1 score.

Project success will be measured by the model’s ability to determine whether a delay will occur or not given a particular station, direction, date, and time.



## Data Collection Plan

### Data Sources

- **MBTA API**
  - Green Line trip updates
  - Car positions
  - Schedule adherence and service alerts

- MBTA Data Hub
- MBTA Travel Times 2026

- **OpenWeather API**
  - Historical and real-time weather data for the Boston area
  - Temperature
  - Precipitation (rain and snow)
  - Wind speed
  - Weather condition labels

### Data Collection Method

- Data will be collected using REST API requests to both the MBTA and OpenWeather APIs.
- MBTA data will be queried historically to compute delay labels.
- Weather data will be aligned with MBTA data using timestamps.
- Processed data will be stored locally in CSV files or a lightweight database for analysis.

## Team & Repository

- Group size: 5 students  
- This repository contains all code, data processing scripts, and analysis notebooks for the project.

**GitHub Repository:**  
https://github.com/araujozBU/CS506-Final-Project

### Team Members

- **Zaki Araujo**  
  araujoz@bu.edu

- **Adrian Dybacki**  
  adybacki@bu.edu

- **Andrew Botolino**  
  botolino@bu.edu

- **Kuba Rozwadowski**  
  kubaroz@bu.edu

- **Rohan Chablani**  
  rohan204@bu.edu
