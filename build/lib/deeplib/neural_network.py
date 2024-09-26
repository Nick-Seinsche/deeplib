import numpy as np

from ad.simple_function import *

class NeuralNetwork:
    def __init__(self, layers, loss_function) -> None:
        self._layers = layers
        self._loss_function = loss_function

    def train(self, data, labels, epochs, learning_rate):
        entry_layer = self._layers[0]
        exit_layer = self._layers[-1]
        _derivatives = np.array([
            self._loss_function.jacobian(
                sum(
                    bias + weight * neuron.value
                    for weight, bias, neuron in zip(
                        exit_layer._input_weights, exit_layer._input_bias, exit_layer._input_neurons
                    )
                )
            )
            * weight * neuron.derivatives[i]
            for i, weight, bias, neuron in enumerate(zip(
                exit_layer._input_weights, exit_layer._input_bias, exit_layer._input_neurons
            ))
        ])
        for epoch in range(epochs):
            for i, (input_data, label) in enumerate(zip(data, labels)):
                entry_layer._neurons[0]._value = input_data
                exit_layer._neurons[0]._value = label

                self.forward_propagate()
                self.backward_propagate()

                for layer in self._layers:
                    for neuron in layer._neurons:
                        for i, (weight, bias, neuron_derivative) in enumerate(zip(
                            neuron._input_weights, neuron._input_bias, neuron._derivatives
                        )):
                            weight -= learning_rate * neuron_derivative * _derivatives[i]
                            bias -= learning_rate * neuron_derivative * _derivatives[i]

                if i % 100 == 0:
                    print(f"Epoch {epoch}, sample {i}, loss {self._loss_function(self._layers[-1]._neurons[0].value)}")
        entry_layer


    def forward_propagate(self):
        for layer in self._layers:
            layer.forward_propagate()

    def backward_propagate(self):
        for layer in reversed(self._layers):
            layer.backward_propagate()


class Layer:
    def __init__(self, number_neurons, neuron_activation_function) -> None:
        self._previous_layer = None
        self._number_neurons = number_neurons
        self._neuron_activation_function = neuron_activation_function
        self._neurons = [
            Neuron(activation_function=neuron_activation_function)
            for _ in range(number_neurons)
        ]

    def connect_to_previous_layer(self, previous_layer):
        self._previous_layer = previous_layer

        for index, neuron in enumerate(self._neurons):
            neuron._input_neurons = previous_layer._neurons
            neuron._derivatives = index * [0] + [1] + [0] * (len(neuron._output_neurons) - index - 1)

    def forward_propagate(self):
        for neuron in self._neurons:
            neuron._forward_propagate()

    def backward_propagate(self):
        for neuron in self._neurons:
            neuron._backward_propagate()


class Neuron:
    def __init__(self, activation_function: SimpleFunction) -> None:
        self._input_neurons = []
        self._input_weights = []
        self._input_bias = []

        self._output_neurons = []
        self._output_weights = []
        self._output_bias = []

        self._value = 0
        self._derivatives = []
        self._activation_function = activation_function

    def _forward_propagate(self):
        if self._input_neurons:
            self._value = self._activation_function(
                sum(
                    bias + weight * neuron.value
                    for weight, bias, neuron in zip(
                        self._input_weights, self._input_bias, self._input_neurons
                    )
                )
            )

    def _backward_propagate(self):
        if self._input_neurons:
            self._derivatives = [
                self._activation_function.jacobian(
                    sum(
                        bias + weight * neuron.value
                        for weight, bias, neuron in zip(
                            self._input_weights, self._input_bias, self._input_neurons
                        )
                    )
                )
                * weight * neuron.derivatives[i]
                for i, weight, bias, neuron in enumerate(zip(
                    self._input_weights, self._input_bias, self._input_neurons
                ))
            ]
