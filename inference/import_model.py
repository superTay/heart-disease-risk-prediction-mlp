"""
Model Import & Inference Module
-------------------------------

This module loads a previously trained neural network model from a JSON file
and exposes a simple API for inference (prediction) on new data.

Typical usage:

    from inference.import_model import HeartDiseaseNNModel

    model = HeartDiseaseNNModel.from_pretrained()
    proba = model.predict_proba(X_new)      # probability of heart disease
    preds = model.predict(X_new)            # 0 = no disease, 1 = disease

This module is designed to be used from a Streamlit app or any other
production-facing interface, without retraining the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from model_components import (
    DenseLayer,
    ActivationReLU,
    ActivationSoftmax,
)


class HeartDiseaseNNModel:
    """
    Wrapper around the from-scratch neural network for clean inference.

    It reconstructs the exact architecture used during training:

        Input (13 features) → Dense(16) → ReLU → Dense(2) → Softmax

    and loads the trained weights and biases from a JSON file exported
    by `export_model.py`.
    """

    def __init__(
        self,
        layer1: DenseLayer,
        activation1: ActivationReLU,
        layer2: DenseLayer,
        activation_output: ActivationSoftmax,
    ):
        self.layer1 = layer1
        self.activation1 = activation1
        self.layer2 = layer2
        self.activation_output = activation_output

    # ------------------------------------------------------------------
    #                CLASSMETHOD: LOAD FROM PRETRAINED JSON
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path] = None,
    ) -> "HeartDiseaseNNModel":
        """
        Load a pretrained model from a JSON file.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to the JSON file exported by export_model.py.
            If None, defaults to: <project_root>/models/heart_disease_nn_model.json

        Returns
        -------
        HeartDiseaseNNModel
            An instance of the model ready for inference.
        """
        if model_path is None:
            project_root = Path(__file__).resolve().parents[1]
            model_path = project_root / "models" / "heart_disease_nn_model.json"

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model JSON not found at: {model_path}. "
                f"Make sure you have run export_model.py."
            )

        with model_path.open("r", encoding="utf-8") as f:
            model_data = json.load(f)

        # --- Parse architecture metadata ---
        arch = model_data["architecture"]
        input_dim = arch["input_dim"]
        hidden_units = arch["hidden_units"]
        output_dim = arch["output_dim"]

        # --- Instantiate layers with correct shapes ---
        layer1 = DenseLayer(n_inputs=input_dim, n_neurons=hidden_units)
        activation1 = ActivationReLU()
        layer2 = DenseLayer(n_inputs=hidden_units, n_neurons=output_dim)
        activation_output = ActivationSoftmax()

        # --- Load weights and biases from JSON ---
        params = model_data["parameters"]

        layer1.weights = np.array(params["layer1"]["weights"], dtype=float)
        layer1.biases = np.array(params["layer1"]["biases"], dtype=float)

        layer2.weights = np.array(params["layer2"]["weights"], dtype=float)
        layer2.biases = np.array(params["layer2"]["biases"], dtype=float)

        # Optional: basic sanity checks
        assert layer1.weights.shape[0] == input_dim, "Layer1 input_dim mismatch"
        assert layer1.weights.shape[1] == hidden_units, "Layer1 hidden_units mismatch"
        assert layer2.weights.shape[0] == hidden_units, "Layer2 input_dim mismatch"
        assert layer2.weights.shape[1] == output_dim, "Layer2 output_dim mismatch"

        print(f"Loaded pretrained model from: {model_path}")
        print(
            f"Architecture: input_dim={input_dim}, hidden_units={hidden_units}, "
            f"output_dim={output_dim}"
        )

        return cls(
            layer1=layer1,
            activation1=activation1,
            layer2=layer2,
            activation_output=activation_output,
        )

    # ------------------------------------------------------------------
    #                INTERNAL: INPUT PREPARATION
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_input(X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Ensure the input is a NumPy array with the expected shape.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features. For this project we expect 13 numeric features
            in the same order as used during training.

        Returns
        -------
        np.ndarray
            2D NumPy array of shape (n_samples, n_features).
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        return X_arr

    # ------------------------------------------------------------------
    #                PUBLIC: PREDICT PROBABILITIES
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Predict class probabilities for new samples.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of probabilities with shape (n_samples, 2),
            where [:, 0] is P(class=0), [:, 1] is P(class=1).
        """
        X_arr = self._prepare_input(X)

        # Forward pass (no dropout during inference)
        self.layer1.forward(X_arr)
        self.activation1.forward(self.layer1.output)
        self.layer2.forward(self.activation1.output)
        self.activation_output.forward(self.layer2.output)

        return self.activation_output.output

    # ------------------------------------------------------------------
    #                PUBLIC: PREDICT CLASS LABELS
    # ------------------------------------------------------------------
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict class labels (0 or 1) for new samples.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input features.
        threshold : float, optional
            Decision threshold applied to the probability of class 1.
            If p(class=1) >= threshold → predict 1, else 0.

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1) of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        # Probability of class 1
        p_class1 = proba[:, 1]

        if threshold == 0.5:
            # Standard argmax decision
            return (p_class1 >= 0.5).astype(int)
        else:
            # Custom threshold on positive class probability
            return (p_class1 >= threshold).astype(int)


if __name__ == "__main__":
    # Quick manual test (for development only)
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "heart_disease_nn_model.json"

    model = HeartDiseaseNNModel.from_pretrained(model_path)

    # Dummy example: use one sample of zeros just to test the pipeline
    dummy_input = np.zeros((1, model.layer1.weights.shape[0]))
    proba = model.predict_proba(dummy_input)
    preds = model.predict(dummy_input)

    print("\nDummy inference test:")
    print("Probabilities:", proba)
    print("Predicted class:", preds)