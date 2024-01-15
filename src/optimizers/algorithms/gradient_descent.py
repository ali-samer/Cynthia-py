import numpy as np


def init_history(params, phistory=None):
    if phistory:
        for key, val in params.items():
            phistory[key].append(val)
        return phistory

    history = {}
    for key, val in params.items():
        history[key] = [val]
    return history


update_history = init_history


def gradient_descent(**kwargs):
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
    params = kwargs['params']
    threshold = kwargs['threshold']
    optimizerObj = kwargs['optimizer_obj']
    iterations = kwargs['iterations']

    history = init_history(params)
    op_obj = optimizerObj(**kwargs)
    print('params:', params.values())
    for i in range(iterations):
        if np.all(np.abs(np.array(list(params.values()))) < threshold):
            print("Gradients are small enough!")
            break
        params = op_obj.update(**kwargs)
        history = update_history(params, phistory=history)
        kwargs['params'] = params

    return history
