# Importing necessary libraries and modules
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from network import NeuralNetwork
from fc_layer import FullyConnectedLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_derivative
from losses import mean_squared_error, mean_squared_error_derivative

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the training data
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype('float32') / 255.0
y_train = to_categorical(y_train)

# Preprocess the test data
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype('float32') / 255.0
y_test = to_categorical(y_test)

# Define the neural network architecture
network = NeuralNetwork()
network.add_layer(FullyConnectedLayer(28 * 28, 100))
network.add_layer(ActivationLayer(tanh, tanh_derivative))
network.add_layer(FullyConnectedLayer(100, 50))
network.add_layer(ActivationLayer(tanh, tanh_derivative))
network.add_layer(FullyConnectedLayer(50, 10))
network.add_layer(ActivationLayer(tanh, tanh_derivative))

# Set the loss function and its derivative
network.set_loss_function(mean_squared_error, mean_squared_error_derivative)

# Train the network on a subset of the training data
num_train_samples = 1000
network.train(x_train[:num_train_samples], y_train[:num_train_samples], epochs=200, learning_rate=0.1)

# Test the network on a subset of the test data
num_test_samples = 500
predictions = network.predict(x_test[:num_test_samples])

# Extract the expected and predicted values
expected_labels = np.argmax(y_test[:num_test_samples], axis=1)
predicted_labels = np.array([np.argmax(prediction) for prediction in predictions])

# Format the expected and predicted values
expected_labels_str = np.array_str(expected_labels, max_line_width=np.inf)
predicted_labels_str = np.array_str(predicted_labels, max_line_width=np.inf)

# Print the expected and predicted values
print("Expected values:")
print(expected_labels_str)
print("Predicted values:")
print(predicted_labels_str)
print()

# Find the indices of incorrect predictions
incorrect_indices = [i for i in range(num_test_samples) if predicted_labels[i] != expected_labels[i]]
num_incorrect_predictions = len(incorrect_indices)

# Summarize the incorrect predictions
if num_incorrect_predictions > 0:
    print(f"Number of incorrect predictions: {num_incorrect_predictions}")
    print("Incorrect prediction details:")
    for idx in incorrect_indices:
        print(f"Index {idx}: Expected {expected_labels[idx]}, Predicted {predicted_labels[idx]}")
else:
    print("All predictions are correct.")