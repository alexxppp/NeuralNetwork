# Neural Network Class
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.loss_derivative = None

    # Add layer to the network
    def add_layer(self, layer):
        self.layers.append(layer)

    # Set the loss function and its derivative
    def set_loss_function(self, loss_function, loss_derivative):
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative

    # Predict output for given input data
    def predict(self, input_data):
        num_samples = len(input_data)
        predictions = []

        # Run network over all samples
        for i in range(num_samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            predictions.append(output)

        return predictions

    # Train the network
    def train(self, x_train, y_train, epochs, learning_rate):
        num_samples = len(x_train)

        # Training loop
        for epoch in range(epochs):
            total_error = 0
            for j in range(num_samples):
                # Forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Compute loss (for display purposes only)
                total_error += self.loss_function(y_train[j], output)

                # Backward propagation
                error = self.loss_derivative(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Calculate average error on all samples
            avg_error = total_error / num_samples
            print('Epoch %d/%d   Error=%f' % (epoch + 1, epochs, avg_error))
