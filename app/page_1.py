# modules/page_dataset_overview.py
"""
Page 1: Dataset Overview
Preview of the cleaned dataset, basic stats, and feature descriptions.
"""

import streamlit as st
import pandas as pd


FEATURE_DESCRIPTIONS = {
    "age": "Age in years.",
    "sex": "Biological sex (1 = male, 0 = female).",
    "cp": "Chest pain type (0–3).",
    "trestbps": "Resting blood pressure (in mm Hg).",
    "chol": "Serum cholesterol (in mg/dl).",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).",
    "restecg": "Resting electrocardiographic results (0–2).",
    "thalach": "Maximum heart rate achieved.",
    "exang": "Exercise induced angina (1 = yes, 0 = no).",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of the peak exercise ST segment (0–2).",
    "ca": "Number of major vessels (0–3) colored by fluoroscopy.",
    "thal": "Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect).",
    "target": "Presence of heart disease (1 = disease, 0 = no disease).",
}


def show():
    """Render the Dataset Overview page."""
    st.header("📁 Dataset Overview")

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.error("No dataset found in session. Please restart the app.")
        return

    df = st.session_state["df"]

    # --- High-level metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of records", f"{len(df):,}")
    col2.metric("Number of features", f"{df.shape[1]}")
    disease_rate = df["target"].mean()
    col3.metric("Disease prevalence", f"{disease_rate*100:.1f}%")

    st.markdown("### 🔍 Sample of the cleaned dataset")
    st.dataframe(df.head(10), width="stretch")

    # --- Basic info / describe ---
    with st.expander("ℹ️ Dataset technical summary"):
        st.write("**Column types**")
        st.write(pd.DataFrame(df.dtypes.astype(str), columns=["dtype"]))

        st.write("**Statistical summary (numeric features)**")
        st.dataframe(df.describe().T, width="stretch")

    # --- Feature dictionary ---
    with st.expander("📚 Feature dictionary (human-readable descriptions)", expanded=True):
        rows = []
        for col in df.columns:
            rows.append(
                {
                    "feature": col,
                    "dtype": str(df[col].dtype),
                    "description": FEATURE_DESCRIPTIONS.get(col, "(no description available)"),
                }
            )
        dict_df = pd.DataFrame(rows)
        st.dataframe(dict_df, width="stretch")