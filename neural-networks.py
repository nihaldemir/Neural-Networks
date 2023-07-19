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

inputs = [1, 2, 3, 2.5]

# every unique input is also going to have unique weight associated with it

weights = [0.2, 0.8, -0.5, 1.0]

# every unique neuron has a unique bias

bias = 2

# 1. for a neuron: add up all the inputs times weights plus bias

"""
 output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] +  inputs[3] * weights[3] + bias

print(output)

"""
# P.2 Coding a Layer

# inputs could either be truly input as in like values from the input layer of a neural network
# which is just going to be a vector of values

# 4 inputs into 3 neurons
# it means there's going to be 3 unique weight sets (w 4 values) and 3 unique biases

weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias2 = 3
bias3 = 0.5

