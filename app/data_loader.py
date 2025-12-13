# modules/data_loader.py
"""
Data loading and basic cleaning utilities for the Heart Disease dataset.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(show_spinner="Loading heart.csv and applying cleaning logic...")
def load_heart_data(path: str = "data/heart.csv") -> pd.DataFrame:
    """
    Load the raw Heart Disease dataset from CSV, apply the same cleaning logic
    used in the training pipeline, and return a clean DataFrame.

    This function is cached to avoid re-loading and re-processing the data
    on every Streamlit rerun.
    """
    df = pd.read_csv(path)

    # Basic sanity: ensure expected columns exist
    expected_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    # Hidden missing values / incorrect encodings (same lógica que en cleaning)
    df["ca"] = df["ca"].replace({4: np.nan})
    df["thal"] = df["thal"].replace({0: np.nan})

    # Impute with mode (categorical-like)
    df["ca"] = df["ca"].fillna(df["ca"].mode()[0])
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

    # Ensure correct dtypes
    int_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                "restecg", "thalach", "exang", "slope", "ca", "thal", "target"]
    float_cols = ["oldpeak"]

    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)

    return df