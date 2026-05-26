# modules/page_data_viz.py
"""
Page 3: Data Exploration
Interactive filters and Plotly charts for the Heart Disease dataset.
"""

import streamlit as st
import pandas as pd
import plotly.express as px


def show():
    """Render the Data Exploration page with filters and interactive charts."""
    st.header("📊 Heart Disease Data Exploration")

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ No data loaded. Please go to the Dataset Overview page first.")
        return

    df = st.session_state["df"]

    # --- Helper to cache unique values ---
    @st.cache_data
    def get_values(column: str):
        return sorted(df[column].dropna().unique().tolist())

    # --- Filters ---
    with st.expander("🎛️ Data Filters", expanded=True):
        with st.form("filter_form"):
            col1, col2 = st.columns(2)

            with col1:
                sex = st.multiselect(
                    "Sex",
                    options=get_values("sex"),
                    format_func=lambda v: "Male" if v == 1 else "Female",
                )
                cp = st.multiselect("Chest Pain Type (cp)", options=get_values("cp"))
                exang = st.multiselect(
                    "Exercise Induced Angina (exang)", options=get_values("exang")
                )
                thal = st.multiselect("Thalassemia (thal)", options=get_values("thal"))

            with col2:
                age_range = st.slider(
                    "Age Range",
                    int(df["age"].min()),
                    int(df["age"].max()),
                    (40, 65),
                )
                chol_range = st.slider(
                    "Cholesterol (chol)",
                    int(df["chol"].min()),
                    int(df["chol"].max()),
                    (int(df["chol"].quantile(0.1)), int(df["chol"].quantile(0.9))),
                )
                thalach_range = st.slider(
                    "Max Heart Rate (thalach)",
                    int(df["thalach"].min()),
                    int(df["thalach"].max()),
                    (int(df["thalach"].quantile(0.1)), int(df["thalach"].quantile(0.9))),
                )

            submitted = st.form_submit_button("✅ Apply filters")

    # --- Apply filters ---
    if submitted:
        df_fil = df.copy()

        if sex:
            df_fil = df_fil[df_fil["sex"].isin(sex)]
        if cp:
            df_fil = df_fil[df_fil["cp"].isin(cp)]
        if exang:
            df_fil = df_fil[df_fil["exang"].isin(exang)]
        if thal:
            df_fil = df_fil[df_fil["thal"].isin(thal)]

        df_fil = df_fil[
            (df_fil["age"].between(age_range[0], age_range[1]))
            & (df_fil["chol"].between(chol_range[0], chol_range[1]))
            & (df_fil["thalach"].between(thalach_range[0], thalach_range[1]))
        ]

        st.session_state["df_fil"] = df_fil

        st.success(f"✅ {len(df_fil):,} patients match the filters.")
        st.dataframe(df_fil.head(10), width="stretch")
    else:
        st.info("Adjust filters and click **Apply filters** to update the view.")

    # --- Visualization section ---
    if "df_fil" in st.session_state and st.session_state["df_fil"] is not None:
        df_fil = st.session_state["df_fil"]

        if df_fil.empty:
            st.warning("Filtered dataset is empty. Please relax your filters.")
            return

        st.markdown("---")
        st.subheader("📈 Interactive Charts")

        col_plot1, col_plot2 = st.columns([4, 1])
        fig_container = col_plot1.empty()

        with col_plot2:
            plot_type = st.selectbox(
                "Plot Type", options=["Scatter", "Histogram", "Box", "Bar"], index=0
            )
            numeric_cols = df_fil.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = df_fil.select_dtypes(include=["int64", "object", "category"]).columns

            x_col = st.selectbox("X-axis", options=numeric_cols)
            if plot_type in ("Scatter", "Box", "Bar"):
                y_col = st.selectbox("Y-axis", options=numeric_cols, index=1)
            else:
                y_col = None

            color_col = st.selectbox(
                "Color",
                options=["None"] + cat_cols.tolist(),
                index=cat_cols.tolist().index("target") + 1 if "target" in cat_cols else 0,
            )

        def generate_plot(df_plot: pd.DataFrame, plot_type: str, x: str, y: str | None, color: str):
            color_arg = color if color != "None" else None

            if plot_type == "Scatter":
                fig = px.scatter(
                    df_plot, x=x, y=y, color=color_arg, opacity=0.8,
                    trendline="ols", hover_data=["age", "chol", "thalach", "target"]
                )
            elif plot_type == "Histogram":
                fig = px.histogram(df_plot, x=x, color=color_arg, nbins=30)
            elif plot_type == "Box":
                fig = px.box(df_plot, x=color_arg or "target", y=y)
            elif plot_type == "Bar":
                fig = px.bar(df_plot, x=x, y=y, color=color_arg)
            else:
                fig = px.scatter(df_plot, x=x, y=y, color=color_arg)

            fig.update_layout(
                template="plotly_white",
                height=500,
                margin=dict(l=20, r=20, t=40, b=40),
                title=dict(text=f"{plot_type} of {y or x} vs {x}", x=0.5),
            )
            return fig

        fig = generate_plot(df_fil, plot_type, x_col, y_col, color_col)
        fig_container.plotly_chart(fig, width="stretch")
    else:
        st.info("No filtered data available. Apply filters to generate charts.")