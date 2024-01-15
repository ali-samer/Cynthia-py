import numpy as np


def gd_adam(df, x, alpha=0.01, beta_1=0.9, beta_2=0.999, iterations=100, epsilon=1e-8):
    history = [x]
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("Adam gradient descent is small enough!")
            break

        grad = df(x)
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * grad ** 2
        # v_1 = v/(1-beta_2)
        t = t + 1
        if True:
            m_1 = m / (1 - np.power(beta_1, t + 1))
            v_1 = v / (1 - np.power(beta_2, t + 1))
        history.append(x)
        return history
