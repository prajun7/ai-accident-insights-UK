import os
import pandas as pd
import numpy as np

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 1 — LOAD DATA
    # ------------------------------------------------------------------
    print("\n--- STEP 1: Load Data ---")
    accidents  = pd.read_csv(os.path.join(DATA_DIR, 'Accidents0515.csv'),  low_memory=False)
    casualties = pd.read_csv(os.path.join(DATA_DIR, 'Casualties0515.csv'), low_memory=False)
    vehicles   = pd.read_csv(os.path.join(DATA_DIR, 'Vehicles0515.csv'),   low_memory=False)

    for name, df in [('Accidents', accidents), ('Casualties', casualties), ('Vehicles', vehicles)]:
        print(f"\n{name}: shape={df.shape}")
        print(df.head(3).to_string())

    # ------------------------------------------------------------------
    # STEP 2 — MERGE
    # ------------------------------------------------------------------
    print("\n--- STEP 2: Merge ---")
    df = accidents.merge(casualties, on='Accident_Index', how='left')
    df = df.merge(vehicles,   on='Accident_Index', how='left')
    print(f"Merged shape: {df.shape}")

    # ------------------------------------------------------------------
    # STEP 3 — DROP HIGH-NULL COLUMNS (>40% missing)
    # ------------------------------------------------------------------
    print("\n--- STEP 3: Drop High-Null Columns ---")
    threshold = 0.4 * len(df)
    null_counts = df.isnull().sum()
    drop_cols = null_counts[null_counts > threshold].index.tolist()
    print(f"Dropped columns (>40% null): {drop_cols}")
    df.drop(columns=drop_cols, inplace=True)

    # ------------------------------------------------------------------
    # STEP 4 — REPLACE INVALID VALUES (-1 → NaN)
    # ------------------------------------------------------------------
    print("\n--- STEP 4: Replace -1 with NaN ---")
    df.replace(-1, np.nan, inplace=True)

    # ------------------------------------------------------------------
    # STEP 5 — IMPUTE REMAINING NULLS
    # ------------------------------------------------------------------
    print("\n--- STEP 5: Impute Nulls ---")
    print(f"Null count before imputation:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)

    print(f"\nNull count after imputation: {df.isnull().sum().sum()}")

    # ------------------------------------------------------------------
    # STEP 6 — REMOVE NULL TARGET
    # ------------------------------------------------------------------
    print("\n--- STEP 6: Remove Null Target ---")
    before = len(df)
    df.dropna(subset=['Accident_Severity'], inplace=True)
    print(f"Dropped {before - len(df)} rows with null Accident_Severity. Remaining: {len(df)}")

    # ------------------------------------------------------------------
    # STEP 7 — FEATURE ENGINEERING
    # ------------------------------------------------------------------
    print("\n--- STEP 7: Feature Engineering ---")

    # Date features
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['Month']      = df['Date'].dt.month
    df['DayOfWeek']  = df['Date'].dt.dayofweek   # 0=Monday, 6=Sunday
    df['IsWeekend']  = (df['DayOfWeek'] >= 5).astype(int)

    # Time features
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
    df['IsNight'] = ((df['Hour'] < 6) | (df['Hour'] >= 20)).astype(int)

    print(f"New columns added: Month, DayOfWeek, IsWeekend, Hour, IsNight")
    print(f"Sample:\n{df[['Date', 'Month', 'DayOfWeek', 'IsWeekend', 'Time', 'Hour', 'IsNight']].head(3).to_string()}")

    # ------------------------------------------------------------------
    # STEP 8 — HANDLE OUTLIERS (cap at 99th percentile)
    # ------------------------------------------------------------------
    print("\n--- STEP 8: Handle Outliers ---")
    for col in ['Number_of_Casualties', 'Speed_limit']:
        if col in df.columns:
            p99 = df[col].quantile(0.99)
            print(f"{col} 99th percentile cap: {p99}")
            df[col] = df[col].clip(upper=p99)

    # ------------------------------------------------------------------
    # STEP 9 — SAVE
    # ------------------------------------------------------------------
    print("\n--- STEP 9: Save ---")
    out_path = os.path.join(OUTPUT_DIR, 'accidents_cleaned.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Final shape: {df.shape}")
    print(f"\nAccident_Severity value counts:\n{df['Accident_Severity'].value_counts()}")

    # ------------------------------------------------------------------
    # STEP 10 — SUMMARY PRINTOUT
    # ------------------------------------------------------------------
    print("\n--- STEP 10: Summary ---")
    print(f"Total rows   : {len(df)}")
    print(f"Total columns: {df.shape[1]}")

    severity_counts = df['Accident_Severity'].value_counts(normalize=True) * 100
    print("\nAccident_Severity distribution:")
    # Actual DfT coding: 1=Fatal, 2=Serious, 3=Slight
    labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    for val, pct in severity_counts.sort_index().items():
        print(f"  {val} ({labels.get(val, val)}): {pct:.1f}%")

    if 'Date' in df.columns:
        print(f"\nDate range: {df['Date'].min().date()} → {df['Date'].max().date()}")


if __name__ == "__main__":
    run()
