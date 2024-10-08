Neural Network and Symbolic Differentiation
===========================================

This repository contains implementations of a neural network for the MNIST dataset and a symbolic differentiation module. The neural network model supports training, saving, and loading of models. The symbolic differentiation module supports operations such as addition, subtraction, multiplication, division, and more, providing a framework for evaluating and differentiating symbolic mathematical expressions.

Features
--------

- **Neural Network for MNIST**:
  - Implemented with support for various layers like `ReLU` and `Softmax`.
  - Supports backpropagation and gradient descent for training.
  - Can save and load models for reuse.

- **Symbolic Differentiation**:
  - Allows symbolic manipulation of variables.
  - Includes operations like `Add`, `Subtract`, `Multiply`, `Divide`, and trigonometric functions like `Sin` and `Cos`.
  - Supports evaluation and differentiation of expressions.

Installation
------------

### Prerequisites

- Python 3.11
- Required libraries (see `pyproject.toml` for package dependencies)

### Steps

1. Clone this repository:

   .. code-block:: bash

      git clone https://github.com/Nick-Seinsche/deeplib.git
      cd repo-name

2. Install dependencies:

   You can manually install the dependencies listed in `pyproject.toml` using `pip`:

   .. code-block:: bash

      pip install -r requirements.txt

Usage
-----

### 1. Neural Network on MNIST

The `neural_network_mnist.py` module trains a neural network on the MNIST dataset. It also supports saving and loading models.

**Example of Training the Model:**

.. code-block:: python

   from neural_network_mnist import NeuralNetwork, LayerRELU, LayerSoftmax, CrossEntropyLoss

   # Define your layers
   l1 = LayerRELU(128, previous_layer=Layer(28 * 28))
   l2 = LayerRELU(64, previous_layer=l1)
   l3 = LayerRELU(32, previous_layer=l2)
   l4 = LayerSoftmax(10, previous_layer=l3)

   # Create a neural network instance
   nn = NeuralNetwork([l1, l2, l3, l4], CrossEntropyLoss())

   # Train the model (data loading code should be added)
   nn.train(train_images, train_labels, epochs=5)

   # Save the model
   nn.save_model("model.pkl")

**Example of Using the Model for Prediction:**

.. code-block:: python

   # Load the trained model
   nn = NeuralNetwork.load_model("model.pkl")

   # Predict an image (reshape the image as needed)
   prediction = nn.predict(test_image)
   print(f"Predicted label: {prediction}")

### 2. Symbolic Differentiation

The `symbolic_differentiation.py` module provides symbolic operations and differentiation.

**Example of Creating and Differentiating Expressions:**

.. code-block:: python

   from deeplib.from_scratch.symbolic_differentiation import *

   x1 = SimpleVariable("x1", (1,))
   x2 = SimpleVariable("x2", (1,))
   variables = [x1, x2]
   term = (
      Sqrt(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4)
      + Exp(x1 + x2 + x3 - x4 * x4)
      + Cos(x1)
      + x2
      + Norm(MatrixVariable("v", (3,), np.array([x1, x2, x3])), 2)
   )

   jterm = term.jacobian(variables)
   hterm = term.hessian(variables)

   f = SimpleFunction("f", variables, term)
   jf = SimpleFunction("jf", variables, jterm)
   hf = SimpleFunction("hf", variables, hterm)

   # evaluate the function
   print(f([1, 2, 3, 4]))
   print(jf([1, 2, 3, 4]))
   print(hf([1, 2, 3, 4]))

License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE.txt>`_ file for details.