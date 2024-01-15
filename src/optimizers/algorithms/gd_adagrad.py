import numpy as np


def gd_adagrad(df, x, alpha=0.01, iterations=100, epsilon=1e-8):
    history = [x]
    gl = np.ones_like(x)
    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("The adagrad gradient is small enough!")
            break;
        grad = df(x)
        gl += grad ** 2
        x -= alpha * grad / (np.sqrt(gl) + epsilon)
        history.append(x)
    return history
