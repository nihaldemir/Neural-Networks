import sys
import numpy as np
import matplotlib

"""
print("Python: ", sys.version)
print("Numpy: ", np.__version__)
print("Matplotlib", matplotlib.__version__)
"""

# P.1 Neuron Code

# there are 4 neurons that are feeding into this neuron that we are gonna build
# neurons are outputting some values

# every unique input is also going to have unique weight associated with it
# every unique neuron has a unique bias

# 1. for a neuron: add up all the inputs times weights plus bias

"""
 output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] +  inputs[3] * weights[3] + bias
"""
# P.2 Coding a Layer

# inputs could either be truly input as in like values from the input layer of a neural network
# which is just going to be a vector of values

# 4 inputs into 3 neurons
# it means there's going to be 3 unique weight sets (w 4 values) and 3 unique biases

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

# P.3 Dot Product

# l = [1,5,6,2] shape = (4,) type: 1D array, Vector
# ll = [[1,5,6,2], [3,2,1,3]] shape = (2,4) type: 2D array, Matrix
# lll = [[[1,5,6,2],[3,2,1,3]], [[5,2,1,2],[6,4,8,4]],[[2,8,5,3],[1,1,9,4]]] shape = (3,2,4) type = 3D array

"""a tensor is an object that can be represented as an array // not just an array but in the context of 
doing deep learning in programming a tensor is represented as an array. """

