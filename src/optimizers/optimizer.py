import numpy as np
import warnings


from src.optimizers.algorithms import *


def get_key_from_params(keys: list, kwargs: dict, callback = None):
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

    This class initializes common parameters used by various optimization algorithms and provides a template for the update method.

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
    def __init__(self, **kwargs):
        self.starting_point_keys = ['x', 'initial_point', 'starting_point']
        self.grad_keys = ['df', 'gradient', 'derivative']
        self.threshold_keys = ['threshold', 'epsilon', 'bar']
        self.learning_rate_keys = ['alpha', 'learning_rate', 'rate']
        self.momentum_keys = ['momentum', 'gamma']

        self.alpha = get_key_from_params(self.learning_rate_keys, kwargs, self.threshold_cb)
        self.initial_input = get_key_from_params(self.starting_point_keys, kwargs, self.starting_point_cb)
        self.grad = get_key_from_params(self.grad_keys, kwargs, self.gradient_func_cb)
        self.df = self.grad
        self.learning_rate = self.alpha

    def update(self, *args):
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

    Adagrad is an algorithm for gradient-based optimization that adapts the learning rate to the parameters. It performs smaller updates for parameters associated with frequently occurring features.

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
        self.squared_gradients = np.ones_like(
            get_key_from_params(self.starting_point_keys, **kwargs))
        self.grad = super.df
        self.init = True
        self.gradient_cumulative = None
        self.multivariate = False


    def update(self, **kwargs):
        """
        Updates the parameters using the Adagrad optimization algorithm.

        Args:
        - kwargs (dict): Keyword arguments containing parameters and gradients.

        Returns:
        - Updated parameters.
        """
        alpha = get_key_from_params(self.learning_rate_keys, kwargs)
        epsilon = get_key_from_params(self.threshold_keys, kwargs, self.threshold_cb)

        if 'params' not in kwargs:
            x = get_key_from_params(self.starting_point_keys, kwargs, self.starting_point_cb)
            grad_of_x = self.df(x)

            self.squared_gradients += grad_of_x ** 2
            return x - alpha * grad_of_x / (np.sqrt(self.squared_gradients) + epsilon)

        self.multivariate = True
        if self.gradient_cumulative is None:
            params = get_key_from_params(['params'], kwargs)
            grads = get_key_from_params(['grads', 'gradient'], kwargs)
            self.gradient_cumulative = {k: np.ones_like(v) for k, v in params.keys()}

        for key, val in params.keys() and self.multivariate:
            grad = grads[key](params)
            self.gradient_cumulative[key] += grad ** 2
            params[key] -= ((alpha * grad) / (np.sqrt(self.gradient_cumulative[key]) + epsilon))
            return params


class Momentum(Optimizers):
    """
    Momentum optimizer.

    This optimizer accelerates the gradient descent algorithm by considering the 'momentum' of the gradients. It helps to accelerate gradients vectors in the right directions, thus leading to faster converging.

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
        self.vector = np.zeros_like(self.initial_input)
        self.gamma = get_key_from_params(self.momentum_keys, kwargs)
        self.init = True
        self.df = super.df

    def update(self, **kwargs):
        """
        Updates the parameters using the Momentum optimization algorithm.

        Args:
        - **kwargs: Arbitrary keyword arguments. Contains parameters for the update step.

        Returns:
        - Updated parameters (numpy.ndarray) based on the momentum optimization.
        """
        x = get_key_from_params(self.starting_point_keys, kwargs)
        self.vector = self.gamma * self.vector + self.alpha * self.df(x)
        return x - self.vector


class Adadelta(Optimizers):
    """
    Adadelta optimizer.

    An extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size.

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
        self.adadelta_keys = ['rho', 'edelta']
        self.x = get_key_from_params(self.starting_point_keys, kwargs, self.starting_point_cb)
        self.rho = get_key_from_params(self.adadelta_keys, kwargs, self.param_cb)
        self.Eg = np.ones_like(self.x)
        self.Edelta = np.ones_like(self.x)

    def update(self, **kwargs):
        """
        Updates the parameters using the Adadelta optimization algorithm.

        Args:
        - **kwargs: Arbitrary keyword arguments. Contains parameters for the update step.

        Returns:
        - Updated parameters (numpy.ndarray) based on the Adadelta optimization.
        """
        x = get_key_from_params(self.starting_point_keys, kwargs, self.starting_point_cb)
        epsilon = get_key_from_params(self.threshold_keys, kwargs, self.threshold_cb)
        alpha = get_key_from_params(self.learning_rate_keys, kwargs, self.learning_rate_cb)

        self.Eg = self.rho*self.Eg + (1-self.rho)*(self.df(x)**2)
        delta = np.sqrt((self.Edelta + epsilon) / (self.Eg + epsilon))*self.df(x)
        self.Edelta = self.rho*self.Eg + (1-self.rho)*(delta**2)
        return x - alpha*delta




