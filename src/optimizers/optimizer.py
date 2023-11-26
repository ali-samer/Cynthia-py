import numpy as np
import warnings


from src.optimizers.algorithms import *


def get_key_from_params(keys: list, kwargs: dict, callback = None):
    for key in keys:
        if key in kwargs and kwargs[key] is not None:
            return kwargs[key]
    return callback() if callback is not None else -1


class Optimizers:
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.squared_gradients = np.ones_like(
            get_key_from_params(self.starting_point_keys, **kwargs))
        self.grad = super.df
        self.init = True
        self.gradient_cumulative = None
        self.multivariate = False


    def update(self, **kwargs):
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

        for key in params.keys() and self.multivariate:
            self.gradient_cumulative[key] += grads[key](params.keys()) ** 2
            params[key] -= (alpha / (np.sqrt(self.gradient_cumulative[key]) + epsilon))
            return params


class Momentum(Optimizers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vector = np.zeros_like(self.initial_input)
        self.gamma = get_key_from_params(self.momentum_keys, kwargs)
        self.init = True
        self.df = super.df

    def update(self, **kwargs):
        x = get_key_from_params(self.starting_point_keys, kwargs)
        self.vector = self.gamma * self.vector + self.alpha * self.df(x)
        return x - self.vector


class Adadelta(Optimizers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adadelta_keys = ['rho', 'edelta']
        self.x = get_key_from_params(self.starting_point_keys, kwargs, self.starting_point_cb)
        self.rho = get_key_from_params(self.adadelta_keys, kwargs, self.param_cb)
        self.Eg = np.ones_like(self.x)
        self.Edelta = np.ones_like(self.x)

    def update(self, **kwargs):
        x = get_key_from_params(self.starting_point_keys, kwargs, self.starting_point_cb)
        epsilon = get_key_from_params(self.threshold_keys, kwargs, self.threshold_cb)
        alpha = get_key_from_params(self.learning_rate_keys, kwargs, self.learning_rate_cb)

        self.Eg = self.rho*self.Eg + (1-self.rho)*(self.df(x)**2)
        delta = np.sqrt((self.Edelta + epsilon) / (self.Eg + epsilon))*self.df(x)
        self.Edelta = self.rho*self.Eg + (1-self.rho)*(delta**2)
        return x - alpha*delta




