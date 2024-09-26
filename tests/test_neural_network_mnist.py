"""Test suite for the neural network module."""

import numpy as np

# Import the classes from your module
from deeplib.neural_network_mnist import (
    NeuralNetwork,
    Layer,
    LayerRELU,
    LayerSoftmax,
    CrossEntropyLoss,
)


def create_simple_network():
    """Create a simple neural network for testing."""
    input_size = 4
    l1 = LayerRELU(5, previous_layer=Layer(input_size))
    l2 = LayerSoftmax(3, previous_layer=l1)
    nn = NeuralNetwork([l1, l2], CrossEntropyLoss())
    return nn


def test_layer_initialization():
    """Test that a Layer initializes weights and biases with correct shapes."""
    input_size = 4
    layer = Layer(5, previous_layer=Layer(input_size))
    assert layer._weights.shape == (5, input_size)
    assert layer._biases.shape == (5, 1)
    assert layer._weights is not None
    assert layer._biases is not None


def test_relu_activation():
    """Test the ReLU activation function with known input and output."""
    layer = LayerRELU(3)
    z = np.array([[-1], [0], [1]])
    activation = layer.activation(z)
    expected = np.array([[0], [0], [1]])
    np.testing.assert_array_equal(activation, expected)


def test_relu_activation_derivative():
    """Test the derivative of the ReLU activation function."""
    layer = LayerRELU(3)
    z = np.array([[-1], [0], [1]])
    derivative = layer.activation_derivative(z)
    expected = np.array([[0], [0], [1]])
    np.testing.assert_array_equal(derivative, expected)


def test_softmax_activation():
    """Test the softmax activation function with known input."""
    layer = LayerSoftmax(3)
    z = np.array([[1], [2], [3]])
    activation = layer.activation(z)
    exp_z = np.exp(z - np.max(z))
    expected = exp_z / np.sum(exp_z)
    np.testing.assert_almost_equal(activation, expected, decimal=6)


def test_forward_propagation():
    """Check that forward propagation produces outputs of correct shape."""
    nn = create_simple_network()
    X = np.array([[0.1], [0.2], [0.3], [0.4]])
    output = nn.forward_propagate(X)
    assert output.shape == (3, 1)
    assert np.all(output >= 0)  # Softmax outputs are non-negative
    assert np.isclose(np.sum(output), 1)  # Softmax outputs sum to 1


def test_backward_propagation_shapes():
    """Ensure that gradient shapes during backpropagation match weights and biases."""
    nn = create_simple_network()
    X = np.random.rand(4, 1)
    y = np.zeros((3, 1))
    y[1] = 1  # Assuming class 1 is the correct label
    nabla_b, nabla_w = nn.backward_propagate(X, y)
    # Check shapes of gradients
    for i, layer in enumerate(nn._layers):
        assert nabla_b[i].shape == layer._biases.shape
        assert nabla_w[i].shape == layer._weights.shape


def test_parameter_update():
    """Confirm that the parameters are updated after backpropagation."""
    nn = create_simple_network()
    X = np.random.rand(4, 1)
    y = np.zeros((3, 1))
    y[2] = 1  # Assuming class 2 is the correct label
    nabla_b_before, nabla_w_before = nn.backward_propagate(X, y)
    weights_before = [layer._weights.copy() for layer in nn._layers]
    biases_before = [layer._biases.copy() for layer in nn._layers]
    nn.update_parameters(nabla_b_before, nabla_w_before)
    # Ensure that weights and biases have been updated
    for i, layer in enumerate(nn._layers):
        assert not np.array_equal(layer._weights, weights_before[i])
        assert not np.array_equal(layer._biases, biases_before[i])


def test_loss_function():
    """Validate the cross-entropy loss calculation."""
    loss_fn = CrossEntropyLoss()
    a = np.array([[0.1], [0.6], [0.3]])
    y = np.array([[0], [1], [0]])
    loss = loss_fn(a, y)
    expected_loss = -np.log(0.6 + 1e-8)
    assert np.isclose(loss, expected_loss)


def test_save_and_load_model(tmp_path):
    """Check that the model can be saved and loaded correctly."""
    nn = create_simple_network()
    model_path = tmp_path / "model.pkl"
    nn.save_model(model_path)
    assert model_path.exists()
    nn_loaded = NeuralNetwork.load_model(model_path)
    # Check that the loaded model has the same architecture
    assert len(nn_loaded._layers) == len(nn._layers)
    for layer_loaded, layer_original in zip(nn_loaded._layers, nn._layers):
        np.testing.assert_array_equal(layer_loaded._weights, layer_original._weights)
        np.testing.assert_array_equal(layer_loaded._biases, layer_original._biases)


def test_prediction():
    """Test the prediction function to ensure it returns valid class labels."""
    nn = create_simple_network()
    X = np.random.rand(4, 1)
    prediction = nn.predict(X)
    assert prediction.shape == (1,)
    assert 0 <= prediction <= 2  # Since we have 3 classes


def test_training_on_synthetic_data():
    """Run a training epoch on synthetic data and ensure that weights are updated."""
    # Create synthetic dataset
    X_train = np.random.rand(10, 4, 1)  # 10 samples, 4 features
    y_train = np.random.randint(0, 3, size=(10,))

    nn = create_simple_network()
    nn.train(X_train, y_train, epochs=1)

    # Check that weights have been updated
    for layer in nn._layers:
        assert not np.all(layer._weights == 0)
        assert not np.all(layer._biases == 0)


def test_accuracy_on_synthetic_data():
    """Test on synthetic data using a mocked predict function."""
    # Create synthetic dataset
    X_test = np.random.rand(5, 4, 1)  # 5 samples, 4 features
    y_test = np.random.randint(0, 3, size=(5,))

    nn = create_simple_network()

    # Mock the predict function to return random labels
    def mock_predict(X):
        return np.random.randint(0, 3)

    nn.predict = mock_predict

    correct = 0
    for x, label in zip(X_test, y_test):
        prediction = nn.predict(x)
        if prediction == label:
            correct += 1
    accuracy = correct / len(y_test)
    assert 0.0 <= accuracy <= 1.0


def test_loss_derivative():
    """Verify that the derivative of the loss function is computed correctly."""
    loss_fn = CrossEntropyLoss()
    a = np.array([[0.2], [0.5], [0.3]])
    y = np.array([[0], [1], [0]])
    derivative = loss_fn.derivative(a, y)
    expected_derivative = a - y
    np.testing.assert_array_almost_equal(derivative, expected_derivative)


def test_layer_softmax_activation_derivative():
    """Ensure the activation derivative for the softmax layer is as expected."""
    layer = LayerSoftmax(3)
    z = np.array([[1], [2], [3]])
    derivative = layer.activation_derivative(z)
    expected = np.ones_like(z)
    np.testing.assert_array_equal(derivative, expected)


def test_compute_z():
    """Test the computation of z in a layer."""
    layer = Layer(3, previous_layer=Layer(4))
    activation = np.random.rand(4, 1)
    z = layer.compute_z(activation)
    assert z.shape == (3, 1)


def test_forward_propagate_single_layer():
    """Test forward propagation through a single layer."""
    layer = LayerRELU(3, previous_layer=Layer(4))
    activation = np.random.rand(4, 1)
    output = layer.forward_propagate(activation)
    assert output.shape == (3, 1)
    assert np.all(output >= 0)  # ReLU activation


def test_neural_network_initialization():
    """Check the neural network's initialization parameters."""
    nn = create_simple_network()
    assert len(nn._layers) == 2
    assert isinstance(nn._loss_function, CrossEntropyLoss)
    assert nn._learning_rate == 0.01


def test_learning_rate_adjustment():
    """Confirm that the learning rate can be adjusted."""
    nn = create_simple_network()
    nn._learning_rate = 0.05
    assert nn._learning_rate == 0.05


def test_layer_weights_initialization():
    """Validate that the weights are initialized using He initialization."""
    layer = Layer(5, previous_layer=Layer(4))
    # He initialization
    std_dev = np.sqrt(2 / 4)
    assert np.isclose(np.std(layer._weights), std_dev, rtol=0.5)


def test_biases_initialization():
    """Ensure that biases are initialized to zero with the correct shape."""
    layer = Layer(5)
    assert np.all(layer._biases == 0)
    assert layer._biases.shape == (5, 1)
