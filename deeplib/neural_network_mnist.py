"""Module for training and testing a neural network on the MNIST dataset."""

from __future__ import annotations
import numpy as np
import pickle
import idx2numpy
from pathlib import Path
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Class representing a neural network model."""

    def __init__(self, layers, loss_function) -> None:
        """Initialize the neural network with layers and a loss function.

        Args:
            layers (list): List of layer instances.
            loss_function (LossFunction): An instance of a loss function class.
        """
        # List of layers
        self._layers = layers
        # Loss function class
        self._loss_function = loss_function
        # Gradient descent step size
        self._learning_rate = 0.01

    def save_model(self, path):
        """Save the neural network model to a file.

        Args:
            path (str): Path to save the model.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path):
        """Load a neural network model from a file.

        Args:
            path (str): Path to the saved model file.

        Returns:
            NeuralNetwork: The loaded neural network instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def forward_propagate(self, X):
        """Perform forward propagation through the network.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Activations from the output layer.
        """
        # Array of activations for each layer
        activations = X
        for layer in self._layers:
            # Propagate to the next layer
            activations = layer.forward_propagate(activations)
        # Return the final activations
        return activations

    def backward_propagate(self, X, y):
        """Perform backward propagation to compute gradients.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels.

        Returns:
            tuple: Gradients for biases and weights.
        """
        # Activation according to the pixel image
        activations = [X]
        zs = []
        activation = X
        for layer in self._layers:
            # Compute the neuron output values depending on its activation
            z = layer.compute_z(activation)
            # Store the z values for backpropagation
            zs.append(z)
            activation = layer.activation(z)
            # Store the activation values for backpropagation
            activations.append(activation)

        # Compute the output error
        output_activation_bar = self._loss_function.derivative(activations[-1], y)

        output_z_value_bar = output_activation_bar * self._layers[
            -1
        ].activation_derivative(zs[-1])

        # Initialize the bar values for the biases and weights
        nabla_b = [np.zeros_like(layer._biases) for layer in self._layers]
        nabla_w = [np.zeros_like(layer._weights) for layer in self._layers]

        # Compute the b_bar and w_bar for the pre-output layer
        nabla_b[-1] = output_z_value_bar
        nabla_w[-1] = np.dot(output_z_value_bar, activations[-2].T)

        # Backpropagate through the rest of the internal layers
        for idx in range(2, len(self._layers) + 1):
            z = zs[-idx]
            sp = self._layers[-idx].activation_derivative(z)
            output_z_value_bar = (
                np.dot(self._layers[-idx + 1]._weights.T, output_z_value_bar) * sp
            )
            nabla_b[-idx] = output_z_value_bar
            nabla_w[-idx] = np.dot(output_z_value_bar, activations[-idx - 1].T)

        return nabla_b, nabla_w

    def update_parameters(self, nabla_b, nabla_w):
        """Update the network parameters using computed gradients.

        Args:
            nabla_b (list): Gradients for biases.
            nabla_w (list): Gradients for weights.
        """
        for i, layer in enumerate(self._layers):
            layer._weights -= self._learning_rate * nabla_w[i]
            layer._biases -= self._learning_rate * nabla_b[i]

    def train(self, X, labels, epochs=10):
        """Train the neural network.

        Args:
            X (numpy.ndarray): Training data.
            labels (numpy.ndarray): Training labels.
            epochs (int, optional): Number of training epochs. Defaults to 10.
        """
        output_size = self._layers[
            -1
        ]._number_neurons  # Get the number of output neurons
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for i, (x, label) in enumerate(zip(X, labels)):
                y = np.zeros((output_size, 1))
                y[label] = 1
                x = x.flatten().reshape(-1, 1)
                nabla_b, nabla_w = self.backward_propagate(x, y)
                self.update_parameters(nabla_b, nabla_w)
            print("Training complete for this epoch.")

    def predict(self, X):
        """Predict the output for given input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            int: Predicted class label.
        """
        activations = self.forward_propagate(X)
        return np.argmax(activations, axis=0)


class Layer:
    """Base class representing a layer in the neural network."""

    def __init__(self, number_neurons, previous_layer: Layer = None) -> None:
        """Initialize the layer with given number of neurons.

        Args:
            number_neurons (int): Number of neurons in the layer.
            previous_layer (Layer, optional): Previous layer instance. Defaults to None.
        """
        self._number_neurons = number_neurons
        self._previous_layer = previous_layer
        self._activation = None
        self._activation_derivative = None
        if previous_layer:
            self._weights = np.random.randn(
                number_neurons, previous_layer._number_neurons
            ) * np.sqrt(2 / previous_layer._number_neurons)
        else:
            self._weights = None
        self._biases = np.zeros((number_neurons, 1))

    def compute_z(self, activation):
        """Compute the weighted input z for the layer.

        Args:
            activation (numpy.ndarray): Activation from the previous layer.

        Returns:
            numpy.ndarray: Weighted input z.
        """
        return np.dot(self._weights, activation) + self._biases

    def forward_propagate(self, activation):
        """Perform forward propagation through the layer.

        Args:
            activation (numpy.ndarray): Activation from the previous layer.

        Returns:
            numpy.ndarray: Activation from this layer.
        """
        z = self.compute_z(activation)
        return self.activation(z)

    def activation(self, z):
        """Activation function for the layer.

        Args:
            z (numpy.ndarray): Weighted input.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    def activation_derivative(self, z):
        """Differentiate the activation function.

        Args:
            z (numpy.ndarray): Weighted input.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError


class LayerRELU(Layer):
    """Layer implementing the ReLU activation function."""

    def activation(self, z):
        """Apply the ReLU activation function.

        Args:
            z (numpy.ndarray): Weighted input.

        Returns:
            numpy.ndarray: Activation after ReLU.
        """
        self._last_z = z
        return np.maximum(0, z)

    def activation_derivative(self, z):
        """Compute the derivative of the ReLU activation function.

        Args:
            z (numpy.ndarray): Weighted input.

        Returns:
            numpy.ndarray: Derivative of ReLU.
        """
        return (z > 0).astype(float)


class LayerSoftmax(Layer):
    """Layer implementing the softmax activation function."""

    def activation(self, z):
        """Apply the softmax activation function.

        Args:
            z (numpy.ndarray): Weighted input.

        Returns:
            numpy.ndarray: Activation after softmax.
        """
        self._last_z = z
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def activation_derivative(self, z):
        """Differentiate the the softmax activation function.

        Args:
            z (numpy.ndarray): Weighted input.

        Returns:
            numpy.ndarray: Derivative of softmax (handled in loss derivative).
        """
        # The derivative of softmax is more complex and is handled in
        # the loss derivative
        return np.ones_like(z)


class LossFunction:
    """Base class for loss functions."""

    def __call__(self, a, y):
        """Compute the loss.

        Args:
            a (numpy.ndarray): Predicted output.
            y (numpy.ndarray): True labels.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    def derivative(self, a, y):
        """Compute the derivative of the loss function.

        Args:
            a (numpy.ndarray): Predicted output.
            y (numpy.ndarray): True labels.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError


class CrossEntropyLoss(LossFunction):
    """Cross-entropy loss function."""

    def __call__(self, a, y):
        """Compute the cross-entropy loss.

        Args:
            a (numpy.ndarray): Predicted output.
            y (numpy.ndarray): True labels.

        Returns:
            float: Computed loss.
        """
        return -np.sum(y * np.log(a + 1e-8))

    def derivative(self, a, y):
        """Compute the derivative of the cross-entropy loss.

        Args:
            a (numpy.ndarray): Predicted output.
            y (numpy.ndarray): True labels.

        Returns:
            numpy.ndarray: Derivative of the loss.
        """
        return a - y


def display_image(images, labels, index):
    """Display an image from the dataset along with its label.

    Args:
        images (numpy.ndarray): Array of images.
        labels (numpy.ndarray): Array of labels corresponding to the images.
        index (int): Index of the image to display.
    """
    image = images[index]
    label = labels[index]
    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")  # Hide axis ticks
    plt.show()


if __name__ == "__main__":
    path = Path("data/MNIST")

    # Load testing images
    test_images_file = "t10k-images.idx3-ubyte"
    test_labels_file = "t10k-labels.idx1-ubyte"

    test_images = idx2numpy.convert_from_file(str(path / test_images_file))
    test_labels = idx2numpy.convert_from_file(str(path / test_labels_file))

    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)

    if not Path("data/MNIST/model.pkl").exists():
        # Load training images
        train_images_file = "train-images.idx3-ubyte"
        train_labels_file = "train-labels.idx1-ubyte"

        # Convert the .ubyte files to numpy arrays
        train_images = idx2numpy.convert_from_file(str(path / train_images_file))
        train_labels = idx2numpy.convert_from_file(str(path / train_labels_file))

        # Check the shape of data
        print("Train images shape:", train_images.shape)
        print("Train labels shape:", train_labels.shape)

        display_image(train_images, train_labels, 3)

        l1 = LayerRELU(128, previous_layer=Layer(28 * 28))
        l2 = LayerRELU(64, previous_layer=l1)
        l3 = LayerRELU(32, previous_layer=l2)
        l4 = LayerSoftmax(10, previous_layer=l3)

        nn = NeuralNetwork([l1, l2, l3, l4], CrossEntropyLoss())
        nn.train(train_images, train_labels, epochs=5)

        nn.save_model("model.pkl")
    nn = NeuralNetwork.load_model("data/MNIST/model.pkl")

    # Testing the model
    correct = 0
    total = len(test_images)
    for i, (x, label) in enumerate(zip(test_images, test_labels)):
        x = x.flatten().reshape(-1, 1) / 255.0
        prediction = nn.predict(x)
        if prediction == label:
            correct += 1
        else:
            # print(f"Prediction: {prediction}, Actual: {label}")
            # display_image(test_images, test_labels, i)
            pass
    print(f"Accuracy: {correct/total * 100}%")
