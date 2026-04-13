#!/usr/bin/env python3
"""Train RandomForest + StandardScaler (same settings as src/classification.py) and save output/7_rf_model.joblib."""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
OUT = os.path.join(OUTPUT_DIR, "7_rf_model.joblib")

FEATURE_COLUMNS = [
    "Speed_limit",
    "Number_of_Vehicles",
    "Vehicle_Manoeuvre",
    "Road_Type",
    "IsNight",
    "Urban_or_Rural_Area",
    "Sex_of_Driver",
    "Junction_Detail",
    "Age_Band_of_Driver",
    "Light_Conditions",
]


def main() -> None:
    x_path = os.path.join(OUTPUT_DIR, "X_final.csv")
    y_path = os.path.join(OUTPUT_DIR, "y_final.csv")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        print(f"Need {x_path} and {y_path} (run src/feature_selection.py first).", file=sys.stderr)
        sys.exit(1)

    print("Loading…")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    print("Training RandomForest (200 trees, class_weight=balanced)…")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "model": model, "feature_columns": FEATURE_COLUMNS},
        OUT,
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
