import warnings

import numpy as np
import sympy as sp

from src.utils.checking import isnumpy
from src.utils.checking import varcheck
from src.utils.policy import policy as pl


def init_aux_vars(numpy_operation, params, dtype=np.float64):
    if not isinstance(params, dict):
        raise Exception("Auxiliary variable must be a dictionary/object")
    if not isinstance(numpy_operation, type(np.ones_like)):
        raise Exception("Initializing operation must be a function from numerical python (numpy) module")
    if not params:
        raise Exception("Parameter dictionary/object is empty. You must include at least"
                        " a single parameter in order to proceed")

    auxiliary_var = {}
    for key, val in params.items():
        # Apply the numpy operation and convert to the specified data type
        auxiliary_var[key] = numpy_operation(val).astype(dtype)

    return auxiliary_var



def get_key_from_params(keys: list, kwargs: dict, callback=None):
    """
    Retrieves a value from a dictionary based on a list of keys.

    Args:
    - keys (list): A list of keys to search in the dictionary.
    - kwargs (dict): The dictionary to search.
    - callback (function, optional): A callback function to execute if none of the keys are found.

    Returns:
    - The value associated with the first key found in the dictionary, or the result of the callback function.
    """
    for key in keys:
        if key in kwargs and kwargs[key] is not None:
            return kwargs[key]
    return callback() if callback is not None else -1


class Optimizers:
    """
    Base class for optimization algorithms.

    This class initializes common parameters used by various optimization algorithms and provides a
    template for the update method.

    Attributes:
    - starting_point_keys (list): Keys to identify the starting point in the arguments.
    - grad_keys (list): Keys to identify the gradient function in the arguments.
    - threshold_keys (list): Keys to identify the threshold value in the arguments.
    - learning_rate_keys (list): Keys to identify the learning rate in the arguments.
    - momentum_keys (list): Keys to identify the momentum value in the arguments.
    - alpha (float): The learning rate.
    - initial_input (float): The initial point for optimization.
    - grad (function): The gradient function.
    - df (function): Alias for the gradient function.
    - learning_rate (float): Alias for the alpha (learning rate).
    """
    starting_point_keys = ['x', 'initial_point', 'starting_point']
    grad_keys = ['df', 'gradient', 'derivative']
    function_keys = ['f', 'y', 'z']
    threshold_keys = ['threshold', 'epsilon', 'bar']
    learning_rate_keys = ['alpha', 'learning_rate', 'rate']
    momentum_keys = ['momentum', 'gamma', 'beta']
    adadelta_keys = ['rho', 'edelta']
    params = ['x', 'y', 'z']

    def __init__(self, **kwargs):
        self.init = True

        self.alpha = get_key_from_params(self.learning_rate_keys, kwargs)
        self.learning_rate = self.alpha
        self.threshold = get_key_from_params(self.threshold_keys, kwargs)
        self.epsilon = self.threshold

        self.cache = {}

        if 'params' not in kwargs:
            raise Exception("function parameters not specified. Must instantiate 'params' variable as a dictionary\n"
                            "Example: { x: 1, y: 1, ... }")

        self.params = kwargs['params']
        self.grads = {}

        for key, val in self.params.items():
            self.cache[key] = sp.symbols(str(key))

        x = self.cache['x']
        y = self.cache['y']
        self.f = x**3 - 3*x*y**2 + 2*y**3 - x**2 + 4*y**2

        print(self.cache)
        print(self.cache.items())
        for key, val in self.params.items():
            partial = sp.diff(self.f, self.cache[key])
            self.grads[key] = sp.lambdify(tuple(self.cache.values()), partial, 'numpy')

    def update(self, **kwargs):
        """
        Template method for updating the optimizer's state.

        This method should be overridden by subclasses.

        Args:
        - args: Variable length argument list.
        """
        pass

    def gradient_func_cb(self):
        raise ValueError("Gradient not specified. Unable to proceed.")

    def starting_point_cb(self):
        raise ValueError("Starting point not specified in arguments.")

    def learning_rate_cb(self):
        warnings.warn("Learning rate not specified. Program will default to a learning rate of float: 0.01")
        return 0.01

    def threshold_cb(self):
        warnings.warn("Threshold not specified. Program will default to a threshold of float: 1e-8")
        return 1e-8


class Adagrad(Optimizers):
    """
    Adagrad optimizer.

    Adagrad is an algorithm for gradient-based optimization that adapts the
    learning rate to the parameters. It performs smaller updates for parameters
    associated with frequently occurring features.

    Attributes:
    - squared_gradients (numpy.ndarray): Squared gradients for each parameter.
    - grad (function): The gradient function.
    - init (bool): Flag to indicate if the optimizer is initialized.
    - gradient_cumulative (dict or None): Cumulative gradients for multivariate cases.
    - multivariate (bool): Flag to indicate if the optimizer is used for multivariate functions.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Adagrad optimizer with the given parameters.

        Args:
        - **kwargs: Arbitrary keyword arguments. Expected to contain the gradient function and initial parameters.
        """
        super().__init__(**kwargs)
        self.squared_grads = init_aux_vars(np.ones_like, self.params)

    def update(self, **kwargs):
        """
        Updates the parameters using the Adagrad optimization algorithm.

        Args:
        - kwargs (dict): Keyword arguments containing parameters and gradients.

        Returns:
        - Updated parameters.
        """

        if not self.init:
            raise Exception("You must initialize object before updating")

        params = kwargs['params']

        for key, val in params.items():
            params_grads = self.grads[key](**params)
            self.squared_grads[key] += params_grads ** 2
            params[key] -= (self.learning_rate * params_grads / np.sqrt(self.squared_grads[key] + self.threshold))

        return params


class Momentum(Optimizers):
    """
    Momentum optimizer.

    This optimizer accelerates the gradient descent algorithm by considering the 'momentum'
    of the gradients. It helps to accelerate gradients vectors in the right directions,
    thus leading to faster converging.

    Attributes:
    - vector (numpy.ndarray): The momentum vector.
    - gamma (float): The momentum coefficient.
    - init (bool): Flag to indicate if the optimizer is initialized.
    - df (function): The derivative function.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Momentum optimizer with the given parameters.

        Args:
        - **kwargs: Arbitrary keyword arguments. Expected to contain the gradient function and initial parameters.
        """
        super().__init__(**kwargs)
        self.velocities = init_aux_vars(np.zeros_like, params=self.params)
        self.momentum = get_key_from_params(self.momentum_keys, kwargs)

    def update(self, **kwargs):
        """
        Updates the parameters using the Momentum optimization algorithm.

        Args:
        - **kwargs: Arbitrary keyword arguments. Contains parameters for the update step.

        Returns:
        - Updated parameters (numpy.ndarray) based on the momentum optimization.
        """
        params = kwargs['params']
        for key, val in params.items():
            v = self.velocities[key]
            self.velocities[key] = self.momentum * v + self.learning_rate * self.grads[key](**params)
            params[key] = val - self.velocities[key]

        return params


class Adadelta(Optimizers):
    """
    Adadelta optimizer.

    An extension of Adagrad that seeks to reduce its aggressive, monotonically
    decreasing learning rate. Instead of accumulating all past squared
    gradients, Adadelta restricts the window of accumulated past gradients to some fixed size.

    Attributes:
    - adadelta_keys (list): Keys to identify Adadelta specific parameters.
    - x (numpy.ndarray): The current position in the parameter space.
    - rho (float): The decay rate.
    - Eg (numpy.ndarray): Running average of the squared gradients.
    - Edelta (numpy.ndarray): Running average of the squared parameter updates.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Adadelta optimizer with the given parameters.

        Args:
        - **kwargs: Arbitrary keyword arguments. Expected to contain the gradient function and initial parameters.
        """
        super().__init__(**kwargs)
        self.rho = get_key_from_params(self.adadelta_keys, kwargs, self.param_cb)
        self.Eg = init_aux_vars(np.ones_like, self.params)
        self.Edelta = init_aux_vars(np.ones_like, self.params)

    def update(self, **kwargs):
        """
        Updates the parameters using the Adadelta optimization algorithm.

        Args:
        - **kwargs: Arbitrary keyword arguments. Contains parameters for the update step.

        Returns:
        - Updated parameters (numpy.ndarray) based on the Adadelta optimization.
        """
        params = kwargs['params']
        for key, val in params.items():
            eval = self.grads[key](**params)
            self.Eg[key] = self.rho * self.Eg[key] + (1 - self.rho) * (eval ** 2)
            delta = np.sqrt((self.Edelta[key] + self.threshold) / (self.Eg[key] + self.threshold)) * eval
            self.Edelta[key] = self.rho * self.Eg[key] + (1 - self.rho) * (delta ** 2)
            params[key] = val - self.learning_rate * delta

        return params


class RMSprop(Optimizers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velocities = init_aux_vars(np.ones_like, self.params)
        self.momentum = get_key_from_params(self.momentum_keys, **kwargs)

    def update(self, **kwargs):
        params = kwargs['params']
        for key, val in params.items():
            grad = self.grads[key](**params)
            v = self.velocities[key]
            v *= self.momentum + (1 - self.momentum) * grad ** 2
            length = np.sqrt(v) + self.threshold
            params[key] -= self.learning_rate * grad / length
            self.velocities[key] = v
        return params
