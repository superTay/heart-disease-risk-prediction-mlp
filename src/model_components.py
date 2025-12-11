import numpy as np
import pandas as pd


class DenseLayer:
    """
    Fully-connected (dense) layer.
    - n_inputs: number of inputs/features coming into the layer
    - n_neurons: number of neurons in this layer
    """

    def __init__(self, n_inputs: int, n_neurons: int):
        # Small random weights and zero biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray):
        """
        Forward pass through the layer.
        Stores inputs and computes output = XW + b
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray):
        """
        Backward pass through the layer.
        Computes gradients w.r.t. weights, biases and inputs.
        """
        # Gradients w.r.t. parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient w.r.t. inputs (to pass to previous layer)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    """
    ReLU activation function: f(x) = max(0, x)
    Used in hidden layers.
    """

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues: np.ndarray):
        # Copy incoming gradient
        self.dinputs = dvalues.copy()
        # Zero gradient where input was negative (neuron inactive)
        self.dinputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    """
    Softmax activation for the output layer.
    Converts raw scores into probabilities that sum to 1.
    """

    def forward(self, inputs: np.ndarray):
        self.inputs = inputs
        # Numerical stability: subtract max per sample
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: np.ndarray):
        """
        Generic backward pass for softmax (using Jacobian).
        Not used when we use the combined Softmax + Cross-Entropy class,
        but good to have for completeness.
        """
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # Flatten output vector
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    """
    Base loss class. Other loss functions will inherit from this.
    """

    def calculate(self, output: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the data loss given model output and true labels.
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossentropy(Loss):
    """
    Categorical Cross-Entropy loss.
    Used when the model outputs a probability distribution over classes
    (e.g. softmax output) and targets are class indices or one-hot vectors.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Number of samples
        n_samples = len(y_pred)

        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # If labels are provided as class indices
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]

        # If labels are one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Loss = -log(correct_class_probability)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        """
        Backward pass for Categorical Cross-Entropy.
        dvalues = softmax output probabilities.
        """
        n_samples = len(dvalues)
        n_labels = len(dvalues[0])

        # If labels are sparse, convert them to one-hot
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]

        # Gradient: -y_true / y_pred
        self.dinputs = -y_true / dvalues
        # Normalize over batch
        self.dinputs = self.dinputs / n_samples


class ActivationSoftmaxLossCategoricalCrossentropy:
    """
    Combined Softmax activation + Categorical Cross-Entropy loss.
    This uses the simplified gradient:
        dL/dZ = (y_pred - y_true) / N
    """

    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()

    def forward(self, inputs: np.ndarray, y_true: np.ndarray) -> float:
        # Softmax activation
        self.activation.forward(inputs)
        self.output = self.activation.output
        # Compute loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        n_samples = len(dvalues)

        # If labels are one-hot, convert to class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy the output
        self.dinputs = dvalues.copy()
        # Subtract 1 from the probabilities of the correct class
        self.dinputs[range(n_samples), y_true] -= 1
        # Normalize over batch
        self.dinputs = self.dinputs / n_samples


class OptimizerAdam:
    """
    Adam optimizer.
    Good default choice for training neural networks on tabular data.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.0,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """
        Should be called before updating parameters in each training step.
        Handles learning rate decay if configured.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer: DenseLayer):
        """
        Updates weights and biases of a given layer using Adam.
        """
        # Initialize momentums and caches if not yet done
        if not hasattr(layer, "weight_momentums"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentums
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums
            + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums
            + (1 - self.beta_1) * layer.dbiases
        )

        # Corrected momentums
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )

        # Update caches (second moment)
        layer.weight_cache = (
            self.beta_2 * layer.weight_cache
            + (1 - self.beta_2) * layer.dweights**2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache
            + (1 - self.beta_2) * layer.dbiases**2
        )

        # Corrected caches
        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )
        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        # Parameter update step
        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        """
        Should be called after all parameter updates in a training step.
        """
        self.iterations += 1

class Dropout:
    """
    Dropout layer for regularization.
    Randomly drops a fraction of neurons during training to prevent overfitting.
    
    Parameters
    ----------
    rate : float
        Fraction of neurons to drop (e.g., 0.2 means drop 20% of neurons).
    """

    def __init__(self, rate: float):
        self.rate = rate

    def forward(self, inputs: np.ndarray, training: bool):
        """
        Forward pass with dropout.
        During training: randomly zeroes neurons and scales remaining ones.
        During inference: dropout disabled (output = inputs).

        Parameters
        ----------
        inputs : np.ndarray
            Activations from the previous layer.
        training : bool
            Whether the model is in training mode.
        """
        self.inputs = inputs

        if not training:
            self.output = inputs
            return

        # Create dropout mask (0 = dropped, 1/(1-rate) = kept)
        self.mask = (np.random.rand(*inputs.shape) > self.rate) / (1 - self.rate)
        self.output = inputs * self.mask

    def backward(self, dvalues: np.ndarray):
        """
        Backward pass through dropout.
        Only neurons that were active during the forward pass receive gradients.
        """
        self.dinputs = dvalues * self.mask