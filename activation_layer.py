# Activation Layer Class
from layer import Layer

# Custom activation layer class
class ActivationLayer(Layer):
    def __init__(self, activation_function, activation_derivative):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    # Perform forward propagation and return the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_function(self.input)
        return self.output

    # Perform backward propagation and return input error
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_derivative(self.input) * output_error