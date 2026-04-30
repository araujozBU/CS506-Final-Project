#Script used to create the ml ready dataset from the raw mbta lr transit time csv files

import os
import shutil
import pandas as pd
import numpy as np
from meteostat import Hourly
from datetime import datetime
from huggingface_hub import hf_hub_download

# Define adjacent station segments
# Format: (from_stop_id, to_stop_id)
adjacent_pairs = [
    # Westbound (Outbound)
    ("70153", "71151"), # Hynes -> Kenmore
    ("71151", "70149"), # Kenmore -> Blandford
    ("70149", "70147"), # Blandford -> BU East
    ("70147", "70145"), # BU East -> BU Central
    ("70145", "170141"),# BU Central -> Amory
    ("170141", "170137"),# Amory -> Babcock
    
    # Eastbound (Inbound)
    ("70134", "170136"), # Packard's -> Babcock
    ("170136", "170140"),# Babcock -> Amory
    ("170140", "70144"), # Amory -> BU Central
    ("70144", "70146"), # BU Central -> BU East
    ("70146", "70148"), # BU East -> Blandford
    ("70148", "71150"), # Blandford -> Kenmore
]

# Weather data fetching function using Meteostat API for Boston Logan Airport (72503)
def get_weather_data(start_dt, end_dt):
    print(f"Fetching weather from {start_dt} to {end_dt}...")
    # Boston Logan ID: 72503
    data = Hourly('72503', start_dt, end_dt)
    
    # 1. Use Meteostat's built-in interpolation to fill gaps (up to 3 hours)
    data = data.interpolate()
    weather = data.fetch().reset_index()
    
    weather['hour'] = weather['time'].dt.hour
    weather['date_str'] = weather['time'].dt.date.astype(str)
    # Keeping: temp, precipitation, and snow depth
    return weather[['date_str', 'hour', 'temp', 'prcp', 'snow']]

# BU Class Surge logic (Based on Appendix A Policy)
def get_surge_flag(dt):
    day = dt.dayofweek  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    hour = dt.hour
    minute = dt.minute
    
    # 1. MWF Standard Pulse (Types A, D, E)
    # Primary End Times: 8:50, 9:55, 11:00, 12:10, 1:25, 2:15, 3:20, 4:25, 5:30
    if day in [0, 2, 4]:
        # Morning Transitions
        if (hour == 8 and 50 <= minute <= 59) or (hour == 9 and 0 <= minute <= 10): return 1 # 8:50 end
        if (hour == 9 and 55 <= minute <= 59) or (hour == 10 and 0 <= minute <= 15): return 1 # 9:55 end
        # Mid-Day Transitions
        if (hour == 11 and 0 <= minute <= 15): return 1 # 11:00 end
        if (hour == 12 and 10 <= minute <= 25): return 1 # 12:10-12:20 end
        # Afternoon Transitions
        if (hour == 13 and 25 <= minute <= 40): return 1 # 1:25 end
        if (hour == 14 and 15 <= minute <= 30): return 1 # 2:15 end
        if (hour == 15 and 20 <= minute <= 35): return 1 # 3:20 end
        if (hour == 16 and 25 <= minute <= 40): return 1 # 4:25 end

    # 2. TR Standard Pulse (Types B, C, D)
    # Primary End Times: 9:15, 10:45, 12:15, 1:45, 3:15, 4:45, 6:15
    elif day in [1, 3]:
        if (hour == 9 and 15 <= minute <= 30): return 1 # 9:15 end
        # THE TRIPLE-WAVE HOTSPOT (Types B, C, and D all end at 10:45)
        if (hour == 10 and 45 <= minute <= 59) or (hour == 11 and 0 <= minute <= 10): return 2 
        if (hour == 12 and 15 <= minute <= 30): return 1 # 12:15 end
        if (hour == 13 and 45 <= minute <= 59) or (hour == 14 and 0 <= minute <= 10): return 1 # 1:45 end
        # THE AFTERNOON HOTSPOT (Type B and D end at 3:15)
        if (hour == 15 and 15 <= minute <= 30): return 2 
        if (hour == 16 and 45 <= minute <= 59) or (hour == 17 and 0 <= minute <= 10): return 1 # 4:45 end
        if (hour == 18 and 15 <= minute <= 30): return 1 # 6:15 end

    return 0

def add_bu_semester_logic(df):
    # 1. Define the Semester Ranges (Start, End)
    # These are the periods where classes are officially "in session"
    semester_ranges = [
        ('2024-01-18', '2024-05-01'), # Spring 2024
        ('2024-09-03', '2024-12-10'), # Fall 2024
        ('2025-01-21', '2025-05-01'), # Spring 2025
        ('2025-09-02', '2025-12-10'), # Fall 2025
        ('2026-01-20', '2026-04-30')  # Spring 2026
    ]

    # 2. Define Specific "No Class" Dates (Holidays & Breaks)
    no_class_dates = [
        '2024-02-19', '2024-04-15', # Spring 24 Holidays
        '2024-10-14', '2024-11-27', '2024-11-28', '2024-11-29', # Fall 24 Holidays
        '2025-02-17', '2025-04-21', # Spring 25 Holidays
        '2025-10-13', '2025-11-26', '2025-11-27', '2025-11-28', # Fall 25 Holidays
        '2026-02-16', '2026-04-20'  # Spring 26 Holidays
    ]
    
    # Add Spring Breaks (inclusive ranges)
    spring_breaks = [
        pd.date_range('2024-03-09', '2024-03-17'),
        pd.date_range('2025-03-08', '2025-03-16'),
        pd.date_range('2026-03-07', '2026-03-15')
    ]
    for break_range in spring_breaks:
        no_class_dates.extend(break_range.strftime('%Y-%m-%d').tolist())

    # 3. Apply the Logic to the Dataframe
    df['from_stop_departure_datetime'] = pd.to_datetime(df['from_stop_departure_datetime'])
    df['date_only'] = df['from_stop_departure_datetime'].dt.strftime('%Y-%m-%d')
    df['day_of_week'] = df['from_stop_departure_datetime'].dt.dayofweek # 0=Mon, 4=Fri
    df['hour'] = df['from_stop_departure_datetime'].dt.hour

    # Check if the date falls in any semester range
    in_semester = pd.Series(False, index=df.index)
    for start, end in semester_ranges:
        in_semester |= df['date_only'].between(start, end)

    # Final Flag: In semester AND NOT a holiday AND is a weekday
    df['is_bu_class_day'] = (
        in_semester & 
        (~df['date_only'].isin(no_class_dates)) & 
        (df['day_of_week'] < 5)
    ).astype(int)

    # 4. Refined "Active Class Hours" Flag (8am - 6pm)
    df['is_active_class_time'] = (
        (df['is_bu_class_day'] == 1) & 
        (df['hour'] >= 8) & 
        (df['hour'] < 18)
    ).astype(int)

    return df

# Main ds creation function
def create_gold_dataset(n_examples=None):
    print("Processing raw CSV files sequentially to save disk space AND maximize speed...")
    
    years = [2024, 2025, 2026]
    months = [f"{i:02d}" for i in range(1, 13)]
    
    pairs_df = pd.DataFrame(adjacent_pairs, columns=['from_stop_id', 'to_stop_id'])
    filtered_chunks = []
    
    repo_id = "adybacki/24_25_26_mbta_lr_travel_times"
    temp_dir = "./temp_csv_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Only load columns we need. This cuts RAM usage enormously and loads files 4x faster.
    cols_to_use = [
        'route_id', 'from_stop_id', 'to_stop_id', 
        'trip_id', 'from_stop_departure_datetime', 'to_stop_arrival_datetime'
    ]
    
    for year in years:
        for month in months:
            # According to HF logs, files end at 2026-02
            if year == 2026 and int(month) > 2:
                continue
                
            filename = f"{year}-{month}_LRTravelTimes.csv"
            print(f"Downloading & Refining {filename}...")
            
            try:
                # Download direct to our controlled temp folder
                file_path = hf_hub_download(
                    repo_id=repo_id, 
                    filename=filename, 
                    repo_type="dataset", 
                    local_dir=temp_dir
                )
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue
                
            # Stream the file directly into a Pandas DataFrame using C-based csv engine
            df_chunk = pd.read_csv(file_path, usecols=cols_to_use, dtype=str)
            
            # Immediately delete the 1GB file to free up the SSD drive completely
            os.remove(file_path)
            
            # Fast Vectorized Filtering 
            df_chunk = df_chunk[df_chunk['route_id'] == 'Green-B'].copy()

            if not df_chunk.empty:
                df_chunk = pd.merge(df_chunk, pairs_df, on=['from_stop_id', 'to_stop_id'], how='inner')
                if not df_chunk.empty:
                    filtered_chunks.append(df_chunk)
                    print(f"   -> Retained {len(df_chunk):,} Green-B segments.")
                    
            if n_examples and sum(len(c) for c in filtered_chunks) >= n_examples:
                break
                
        if n_examples and sum(len(c) for c in filtered_chunks) >= n_examples:
            break
            
    # Clean up empty temp folder
    shutil.rmtree(temp_dir, ignore_errors=True)

    if not filtered_chunks:
        print("No matching rows found.")
        return pd.DataFrame()

    print("Concatenating all filtered months...")
    df = pd.concat(filtered_chunks, ignore_index=True)
    
    print(f"Filtered down to {len(df):,} relevant rows. Processing datetimes...")
    
    df['from_stop_departure_datetime'] = pd.to_datetime(df['from_stop_departure_datetime'])
    df['to_stop_arrival_datetime'] = pd.to_datetime(df['to_stop_arrival_datetime'])
    
    # Calculate Dwell at origin (time spent at each segment's departure station)
    df = df.sort_values(by=['trip_id', 'from_stop_departure_datetime'])
    # Shift arrival times to align with current segment's origin
    df['prev_arrival_at_origin'] = df.groupby('trip_id')['to_stop_arrival_datetime'].shift(1)
    # Dwell = departure time from origin - arrival time at origin (from previous segment)
    df['dwell_time_sec'] = (df['from_stop_departure_datetime'] - df['prev_arrival_at_origin']).dt.total_seconds()
    
    # Clean Outliers
    df = df[
        df['dwell_time_sec'].isna() |
        ((df['dwell_time_sec'] >= 0) & (df['dwell_time_sec'] < 600))
    ].copy()
    
    # Drop temporary helper columns
    df = df.drop(columns=['prev_arrival_at_origin'], errors='ignore')
    
    # Add BU Semester & Class Time Logic
    df = add_bu_semester_logic(df)

   # Only apply the surge logic if it's actually a day when classes are in session
    df['is_student_surge'] = (
        (df['is_bu_class_day'] == 1) & 
        (df['from_stop_departure_datetime'].apply(get_surge_flag) == 1)
    ).astype(int)
    
    # Get and Join Weather
    start_dt = df['from_stop_departure_datetime'].min().tz_localize(None)
    end_dt = df['from_stop_departure_datetime'].max().tz_localize(None)
    weather = get_weather_data(start_dt, end_dt)
    
    # Rename date_only to date_str so it matches weather dataframe
    df['date_str'] = df['date_only']
    
    final_df = pd.merge(df, weather, on=['date_str', 'hour'], how='left')
    
    # DO THE FILLING AFTER THE MERGE
    # 1. Fill snow with 0 if it's entirely missing (since API omits when there is no snow)
    final_df['snow'] = final_df['snow'].fillna(0)
    
    # 2. Backfill and forward-fill any missing temp/prcp from the join (missing hours)
    final_df[['temp', 'prcp']] = final_df[['temp', 'prcp']].bfill().ffill()
    
    # 3. Round weather values to clean up floating point errors
    final_df['temp'] = final_df['temp'].round(1)
    final_df['prcp'] = final_df['prcp'].round(2)
    final_df['snow'] = final_df['snow'].round(2)
    
    # Keep terminal segments even when dwell cannot be computed.
    # Downstream training fills missing dwell values.
    return final_df

# Execution and ds upload
# Set n_examples to None to comb through the entire dataset
df_gold = create_gold_dataset(n_examples=None)

# Save locally as Parquet (smaller and preserves types better than CSV)
filename = "bu_green_line_gold.parquet"
df_gold.to_parquet(filename)
print(f"Dataset created with {len(df_gold)} rows.")