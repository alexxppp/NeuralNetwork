import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the training data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32') / 255.0
y_train = to_categorical(y_train)

# Preprocess the test data
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32') / 255.0
y_test = to_categorical(y_test)

# Define the neural network architecture
net = Network()
net.add(FCLayer(28*28, 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))

# Set the loss function and its derivative
net.use(mse, mse_prime)

# Train the network on a subset of the training data
num_train_samples = 1000
net.fit(x_train[:num_train_samples], y_train[:num_train_samples], epochs=10, learning_rate=0.1)

# Test the network on a subset of the test data
num_test_samples = 500
out = net.predict(x_test[:num_test_samples])

# Extract the expected and predicted values
expected_values = np.argmax(y_test[:num_test_samples], axis=1)
predicted_values = np.array([np.argmax(o) for o in out])  # Convert to numpy array

# Format the expected and predicted values
expected_str = np.array_str(expected_values, max_line_width=np.inf)
predicted_str = np.array_str(predicted_values, max_line_width=np.inf)

# Print the expected and predicted values
print("Expected values:")
print(expected_str)
print("Predicted values:")
print(predicted_str)
print()

# Find the indices of incorrect predictions
incorrect_indices = [i for i in range(num_test_samples) if predicted_values[i] != expected_values[i]]
num_errors = len(incorrect_indices)

# Summarize the incorrect predictions
if num_errors > 0:
    print(f"Number of incorrect predictions: {num_errors}")
    print("Incorrect prediction details:")
    for idx in incorrect_indices:
        print(f"Index {idx}: Expected {expected_values[idx]}, Predicted {predicted_values[idx]}")
else:
    print("All predictions are correct.")



import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)



class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))




            
import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;



# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    




    from layer import Layer
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of neurons of the input
    # output_size = number of neurons of the output
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    



    import numpy as np

# Function to compute the hyperbolic tangent activation
def tanh(x):
    """
    Compute the hyperbolic tangent activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array after applying the hyperbolic tangent activation.
    """
    return np.tanh(x)

# Function to compute the derivative of the hyperbolic tangent activation
def tanh_prime(x):
    """
    Compute the derivative of the hyperbolic tangent activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array after applying the derivative of the hyperbolic tangent activation.
    """
    return 1 - np.tanh(x) ** 2




from layer import Layer

# Custom activation layer class
class ActivationLayer(Layer):
    def __init__(self, activation_func, activation_prime_func):
        self.activation_func = activation_func
        self.activation_prime_func = activation_prime_func

    # Perform forward propagation and return the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_func(self.input)
        return self.output

    # Perform backward propagation and return input error
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime_func(self.input) * output_error
