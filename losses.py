# Loss Functions
import numpy as np

# Mean Squared Error (MSE) loss function and its derivative
def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mean_squared_error_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
