
# Activation Functions
import numpy as np

# Hyperbolic tangent activation function and its derivative
def tanh(x):
    """
    Compute the hyperbolic tangent activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array after applying the hyperbolic tangent activation.
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Compute the derivative of the hyperbolic tangent activation function.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Output array after applying the derivative of the hyperbolic tangent activation.
    """
    return 1 - np.tanh(x) ** 2
