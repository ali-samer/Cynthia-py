from typing import Callable, List
import numpy as np

def gradient_descent(df: Callable[[np.ndarray], np.ndarray],
                     x: np.ndarray,
                     learning_rate: float = 0.01,
                     iterations: int = 100,
                     threshold: float = 1e-8,
                     optimizerObj: 'Optimizer' = None,
                     callback: Callable[[np.ndarray, int], None] = None) -> List[np.ndarray]:
    """
    Performs gradient descent optimization.

    :param df: The derivative of the function to be minimized.
    :param x: Initial value for the parameter to be optimized.
    :param learning_rate: Step size for each iteration.
    :param iterations: Maximum number of iterations.
    :param threshold: Threshold for stopping criteria based on the gradient's magnitude.
    :param optimizerObj: Custom optimizer object, must have an 'update' method.
    :param callback: Optional callback function for custom operations at each iteration.
    :return: History of parameter values through iterations.
    """
    history = [x]
    for i in range(iterations):
        grad = df(x)
        if np.linalg.norm(grad) < threshold:
            break

        x = optimizerObj.update(df, x, learning_rate, threshold) if optimizerObj else x - learning_rate * grad
        history.append(x)

        if callback:
            callback(x, i)

    return history
