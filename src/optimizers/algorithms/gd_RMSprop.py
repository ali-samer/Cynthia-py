import numpy as np


def gd_RMSprop(df, x, alpha=0.01, beta=0.9, iterations=100, epsilon=1e-8):
    history = [x]
    v = np.ones_like(x)

    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("RMS prop gradient descent is small enough!")
            break

        grad = df(x)
        v = beta * v + (1 - beta) * grad ** 2
        x -= alpha * grad / (np.sqrt(v) + epsilon)
        history.append(x)

    return history
