
"""
Model Training Module
---------------------

This script trains the neural network on the full dataset using a simple
full-batch gradient descent loop. The architecture includes:

- Dense hidden layer with ReLU activation
- Dropout layer for regularization
- Dense output layer with Softmax activation
- Combined Softmax + Categorical Cross-Entropy loss
- Adam optimizer

The goal of this module is to validate that the core training pipeline
(forward, loss, backward, and parameter updates) runs correctly and
converges on the Heart Disease dataset.
"""

import numpy as np

from data_cleaning import X, y
from model_components import (
    DenseLayer,
    ActivationReLU,
    Dropout,
    ActivationSoftmaxLossCategoricalCrossentropy,
    OptimizerAdam,
)

# Convert pandas objects to NumPy arrays
X = X.values
y = y.values

# Basic metadata
n_features = X.shape[1]   # should be 13
n_classes = 2             # binary classification (0 / 1)
epochs = 500

print(f"Training dataset shape -> X: {X.shape}, y: {y.shape}")
print(f"Features: {n_features}, Classes: {n_classes}")

# ------------------------------------------------------------------
#               MODEL INSTANTIATION (WITH DROPOUT)
# ------------------------------------------------------------------

layer1 = DenseLayer(n_inputs=n_features, n_neurons=16)
activation1 = ActivationReLU()

# Dropout layer (regularization):
# Randomly deactivates a fraction of neurons during training to prevent overfitting.
dropout1 = Dropout(rate=0.2)

layer2 = DenseLayer(n_inputs=16, n_neurons=n_classes)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = OptimizerAdam(learning_rate=0.001, decay=1e-3)

# ------------------------------------------------------------------
#          SINGLE TRAINING STEP (FORWARD + BACKWARD)
# ------------------------------------------------------------------

# Forward pass
layer1.forward(X)
activation1.forward(layer1.output)
dropout1.forward(activation1.output, training=True)
layer2.forward(dropout1.output)
loss = loss_activation.forward(layer2.output, y)

# Predictions and accuracy
predictions = np.argmax(loss_activation.output, axis=1)
accuracy = np.mean(predictions == y)

print("\nSingle training step (before epoch loop):")
print(f"Initial loss: {loss:.4f}")
print(f"Initial accuracy: {accuracy:.4f}")

# Backward pass
loss_activation.backward(loss_activation.output, y)
layer2.backward(loss_activation.dinputs)
dropout1.backward(layer2.dinputs)
activation1.backward(dropout1.dinputs)
layer1.backward(activation1.dinputs)

# Parameter update
optimizer.pre_update_params()
optimizer.update_params(layer1)
optimizer.update_params(layer2)
optimizer.post_update_params()

print("Single forward/backward pass completed and parameters updated.\n")

# ------------------------------------------------------------------
#                     FULL TRAINING LOOP
# ------------------------------------------------------------------

print("Starting epoch training loop...\n")

for epoch in range(epochs):
    # Forward pass (training mode)
    layer1.forward(X)
    activation1.forward(layer1.output)
    dropout1.forward(activation1.output, training=True)
    layer2.forward(dropout1.output)
    loss = loss_activation.forward(layer2.output, y)

    # Predictions & accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    dropout1.backward(layer2.dinputs)
    activation1.backward(dropout1.dinputs)
    layer1.backward(activation1.dinputs)

    # Update parameters
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()

    # Logging
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | "
            f"LR: {optimizer.current_learning_rate:.6f}"
        )

print("\nTraining loop completed.")