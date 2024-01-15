import numpy as np


def gd_adadelta(df, x, alpha=0.1, rho=0.9, iterations=100, epsilon=1e-8):
    history = [x]
    Eg = np.ones_like(x)
    Edelta = np.ones_like(x)

    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("AdaDelta gradient descent is small enough!")
            break

        grad = df(x)
        Eg = rho * Eg + (1 - rho) * (grad ** 2)
        delta = np.sqrt((Edelta + epsilon) / (Eg + epsilon)) * grad
        x = x - alpha * delta
        Edelta = rho * Edelta + (1 - rho) * (delta ** 2)
        history.append(x)

    return history
