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

    def update(self, **kwargs):
        x = get_key_from_params(self.starting_point_keys, kwargs, self.starting_point_cb)
        alpha = get_key_from_params(self.learning_rate_keys, kwargs)
        grad_of_x = self.grad(x)
        epsilon = get_key_from_params(self.threshold_keys, kwargs, self.threshold_cb)
        self.squared_gradients += grad_of_x ** 2
        return x - alpha * grad_of_x / (np.sqrt(self.squared_gradients) + epsilon)


class Momentum(Optimizers):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vector = np.zeros_like(self.initial_input)
        self.gamma = get_key_from_params(self.momentum_keys, kwargs)
        self.init = True
        self.df = super.df
        self.grad = super.grad

    def update(self, **kwargs):
        x = get_key_from_params(self.starting_point_keys, kwargs)
        self.vector = self.gamma * self.vector + self.alpha * self.df(x)
        return x - self.vector


class Adadelta(Optimizers):
    def __init__(self, **kwargs):
        super.__init__(**kwargs)

