import numpy as np

from src.optimizers.paramOptimizer.gradient_descent_optimizer import gd_optimizer
from src.optimizers.paramOptimizer.optimizer import SGD
from src.optimizers.paramOptimizer.optimizer import SGD_Momentum

f = lambda x, y: (x ** 2 / 16) + (9 * y ** 2)

df = lambda x: np.array(((1 / 8) * x[0], 18 * x[1]))
x0 = np.array([-2.4, 0.2])
optimizator = SGD([x0], 0.1)
path = gd_optimizer(df, optimizator, 100)
print(path[-1])
path = np.asarray(path)
path = path.transpose()

##### now switching to SGD_Momentum
x0 = np.array([-2.4, 0.2])
optimizator = SGD_Momentum([x0], 0.1, 0.8)
path = gd_optimizer(df, optimizator, 1000)
print(path[-1])

path = np.asarray(path)
path = path.transpose()