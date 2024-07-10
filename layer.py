# Base Layer Class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Compute the output of a layer for a given input
    def forward_propagation(self, input_data):
        raise NotImplementedError

    # Compute the gradient of the error with respect to the input
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
