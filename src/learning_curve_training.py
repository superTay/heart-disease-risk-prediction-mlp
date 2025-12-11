import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model_components import (
    DenseLayer,
    ActivationReLU,
    Dropout,
    ActivationSoftmaxLossCategoricalCrossentropy,
    OptimizerAdam
)
from data_cleaning import X, y

# Convert to NumPy arrays
X = X.values
y = y.values

# 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

n_features = X_train.shape[1]
n_classes = 2
epochs = 300

# Model definition
layer1 = DenseLayer(n_inputs=n_features, n_neurons=16)
activation1 = ActivationReLU()
dropout1 = Dropout(rate=0.2)
layer2 = DenseLayer(n_inputs=16, n_neurons=n_classes)
loss_activation = ActivationSoftmaxLossCategoricalCrossentropy()
optimizer = OptimizerAdam(learning_rate=0.001, decay=1e-3)

# Storage for learning curves
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("Starting learning-curve training...\n")

for epoch in range(epochs):

    # ---- FORWARD PASS (TRAIN) ----
    layer1.forward(X_train)
    activation1.forward(layer1.output)
    dropout1.forward(activation1.output, training=True)
    layer2.forward(dropout1.output)
    train_loss = loss_activation.forward(layer2.output, y_train)

    train_pred = np.argmax(loss_activation.output, axis=1)
    train_acc = np.mean(train_pred == y_train)

    # ---- BACKWARD PASS ----
    loss_activation.backward(loss_activation.output, y_train)
    layer2.backward(loss_activation.dinputs)
    dropout1.backward(layer2.dinputs)
    activation1.backward(dropout1.dinputs)
    layer1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()

    # ---- VALIDATION PASS ----
    layer1.forward(X_val)
    activation1.forward(layer1.output)
    dropout1.forward(activation1.output, training=False)
    layer2.forward(dropout1.output)
    val_loss = loss_activation.forward(layer2.output, y_val)

    val_pred = np.argmax(loss_activation.output, axis=1)
    val_acc = np.mean(val_pred == y_val)

    # ---- STORE METRICS ----
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch}: "
            f"Train loss={train_loss:.4f}, val loss={val_loss:.4f}, "
            f"train acc={train_acc:.4f}, val acc={val_acc:.4f}"
        )


# ------------------------------------------------------------------
#           SAVE LEARNING CURVES FOR STREAMLIT LATER
# ------------------------------------------------------------------

np.savez(
    "learning_curves.npz",
    train_losses=np.array(train_losses),
    val_losses=np.array(val_losses),
    train_accuracies=np.array(train_accuracies),
    val_accuracies=np.array(val_accuracies),
)

print("\nLearning curves saved to learning_curves.npz")


# ------------------------------------------------------------------
#            OPTIONAL: PLOT LEARNING CURVES HERE
# ------------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Learning Curve - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Learning Curve - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()