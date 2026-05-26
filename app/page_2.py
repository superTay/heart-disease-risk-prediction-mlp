# modules/page_model_training.py
"""
Page 2: Model Training & Validation
Displays learning curves and ROC-AUC curves from offline training artifacts.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


@st.cache_data(show_spinner="Loading learning curves...")
def load_learning_curves(path: str = "learning_curves.npz") -> dict:
    data = np.load(path)
    return {
        "train_losses": data["train_losses"],
        "val_losses": data["val_losses"],
        "train_accuracies": data["train_accuracies"],
        "val_accuracies": data["val_accuracies"],
    }


@st.cache_data(show_spinner="Loading ROC curve data...")
def load_roc_curves(path: str = "roc_curves.npz") -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "fprs": data["fprs"],
        "tprs": data["tprs"],
        "aucs": data["aucs"],
    }


def show():
    """Render the Model Training & Validation page."""
    st.header("🧠 Model Training & Validation")

    # --- Learning Curves ---
    try:
        curves = load_learning_curves()
    except FileNotFoundError:
        st.error("learning_curves.npz not found. Please run the training script first.")
        return

    st.subheader("📉 Learning Curves (Loss & Accuracy)")

    epochs = list(range(len(curves["train_losses"])))

    # Loss
    fig_loss = go.Figure()
    fig_loss.add_trace(
        go.Scatter(x=epochs, y=curves["train_losses"], mode="lines", name="Train Loss")
    )
    fig_loss.add_trace(
        go.Scatter(x=epochs, y=curves["val_losses"], mode="lines", name="Validation Loss")
    )
    fig_loss.update_layout(
        title="Train vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_loss, width="stretch")

    # Accuracy
    fig_acc = go.Figure()
    fig_acc.add_trace(
        go.Scatter(x=epochs, y=curves["train_accuracies"], mode="lines", name="Train Accuracy")
    )
    fig_acc.add_trace(
        go.Scatter(x=epochs, y=curves["val_accuracies"], mode="lines", name="Validation Accuracy")
    )
    fig_acc.update_layout(
        title="Train vs Validation Accuracy",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_acc, width="stretch")

    st.markdown(
        """
The learning curves suggest that the model is **learning in a stable way** and that
validation performance closely tracks training performance, which is consistent with
a well-regularized neural network (Dropout + Adam with decay).
        """
    )

    # --- ROC Curves ---
    st.markdown("---")
    st.subheader("📈 ROC Curves (5-Fold Cross-Validation)")

    try:
        roc_data = load_roc_curves()
    except FileNotFoundError:
        st.error("roc_curves.npz not found. Please run the evaluation script first.")
        return

    fprs = roc_data["fprs"]
    tprs = roc_data["tprs"]
    aucs = roc_data["aucs"]

    fig_roc = go.Figure()

    for i, (fpr, tpr, auc) in enumerate(zip(fprs, tprs, aucs), start=1):
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"Fold {i} (AUC={auc:.2f})",
            )
        )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash", color="gray"),
        )
    )

    fig_roc.update_layout(
        title="ROC Curves per Fold",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig_roc, width="stretch")

    st.markdown(
        """
The ROC curves and AUC values provide a **global measure of discriminative power**.
Values clearly above 0.8 indicate that the model is able to separate positive and
negative cases with good reliability across cross-validation folds.
        """
    )