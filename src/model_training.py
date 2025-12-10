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

# Note:
# Before implementing the full training loop (multiple epochs),
# we first run a single forward + backward pass on the whole dataset.
# This helps validate:
# - that all layers are connected correctly,
# - that shapes are consistent,
# - that the loss and accuracy can be computed,
# - and that gradients flow without numerical issues.

# --- Single training step (forward + loss + accuracy + backward + update) ---

# Forward pass
# 1) First dense layer
layer1.forward(X)
activation1.forward(layer1.output)

# 2) Output layer
layer2.forward(activation1.output)

# 3) Softmax + Categorical Cross-Entropy combined
loss = loss_activation.forward(layer2.output, y)  # we can pass class indices (0/1)

# Predictions and accuracy
predictions = np.argmax(loss_activation.output, axis=1)
accuracy = np.mean(predictions == y)

print("\nSingle training step (before loop):")
print(f"Initial loss: {loss:.4f}")
print(f"Initial accuracy: {accuracy:.4f}")

# Backward pass
# 1) Softmax + CCE combined backward
loss_activation.backward(loss_activation.output, y)

# 2) Backprop through last dense layer
layer2.backward(loss_activation.dinputs)

# 3) Backprop through ReLU
activation1.backward(layer2.dinputs)

# 4) Backprop through first dense layer
layer1.backward(activation1.dinputs)

# Parameter update with Adam
optimizer.pre_update_params()
optimizer.update_params(layer1)
optimizer.update_params(layer2)
optimizer.post_update_params()

print("Single forward/backward pass completed and parameters updated.")

# ----------------------------------------------------------
#               TRAINING LOOP (FULL BATCH)
# ----------------------------------------------------------

# Number of epochs for training
epochs = 500

print("\nStarting training loop...\n")

for epoch in range(epochs):

    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    loss = loss_activation.forward(layer2.output, y)

    # Predictions and accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # Update parameters
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()

    # Print training progress every 50 epochs
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | "
            f"LR: {optimizer.current_learning_rate:.6f}"
        )