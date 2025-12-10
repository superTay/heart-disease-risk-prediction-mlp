import numpy as np

# Import model components (layers, activations, loss, optimizer)
from model_components import (
    DenseLayer,
    ActivationReLU,
    ActivationSoftmaxLossCategoricalCrossentropy,
    OptimizerAdam,
)

# Import preprocessed data (features X and labels y)
from data_cleaning import X, y


# Basic metadata about the dataset
n_features = X.shape[1]      # should be 13 for this dataset
n_classes = 2                # binary classification: 0 or 1


# Instantiate model architecture
# Hidden layer: 16 neurons, ReLU activation
layer1 = DenseLayer(n_inputs=n_features, n_neurons=16)
activation1 = ActivationReLU()

# Output layer: 2 neurons (one per class), Softmax + Categorical Cross-Entropy
layer2 = DenseLayer(n_inputs=16, n_neurons=n_classes)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()

# Optimizer: Adam (good default for this kind of problem)
optimizer = OptimizerAdam(
    learning_rate=0.001,
    decay=1e-3
)

print("Model components instantiated successfully.")
print(f"Input features: {n_features}, Output classes: {n_classes}")


# --- One-Hot Encoding for labels (y) ---
# Convert integer labels (0 or 1) into one-hot encoded vectors
y_one_hot = np.eye(n_classes)[y]

print("One-hot encoding completed. Sample:")
print(y[:5], "→", y_one_hot[:5])