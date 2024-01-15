import numpy as np


def gd_mntm(df, x, alpha=0.01, gamma=0.8, iterations=100, epsilon=1e-8):
    history = [x]
    v = np.zeros_like(x)

    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("The momentum gradient is small enough!")
            break
        v = gamma * v + alpha * df(x)
        x -= v
        history.append(x)

    return history
