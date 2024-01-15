import matplotlib.pyplot as plt
import numpy as np


def loss(x, y, w, b):
    pY = w * x + b
    return np.mean((pY - y) ** 2) / 2


def plot_loss_history(x, y, w, b, history, fromTo=(4, 6)):
    costs = [loss(x, y, w, b) for w, b in history]
    plt.axis([0, len(costs), fromTo[0], fromTo[1]])
    plt.plot(costs)
