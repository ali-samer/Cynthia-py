import numpy as np


def gd_optimizer(df, optimizer, iterations, epsilon=1e-8):
    x, = optimizer.parameters()
    x = x.copy()
    history = [x]
    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("The gradient optimizer is small enough!")
            break
        grad = df(x)
        x, = optimizer.step([grad])
        x = x.copy()
        history.append(x)
    return history
