from neural_network import *
from ad.simple_function import *

l1 = Layer(number_neurons=30,
           neuron_activation_function=RELU)

l2 = Layer(number_neurons=10,
           neuron_activation_function=RELU)
l2.connect_to_previous_layer(l1)

l3 = Layer(number_neurons=10,
           neuron_activation_function=SoftMax(num_inputs=10, num_outputs=10))
l3.connect_to_previous_layer(l2)


nn = NeuralNetwork(layers=[l1, l2, l3],
                   loss_function=None)