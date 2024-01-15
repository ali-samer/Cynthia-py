import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import src.optimizers.algorithms as alg
from src.utils.plot import plot_path

f = lambda x, y: (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2
minima = np.array([3., .5])
minima_ = minima.reshape(-1, 1)

xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2

x_list = np.arange(xmin, xmax + xstep, xstep)
y_list = np.arange(ymin, ymax + ystep, ystep)

x, y = np.meshgrid(x_list, y_list)
z = f(x, y)

fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=10, azim=-10)
ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1,
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)

ax.plot(*minima_, f(*minima_), 'r*', markersize=10)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

# plt.draw()

df_x = lambda x, y: 2 * (1.5 - x + x * y) * (y - 1) + 2 * (2.25 - x + x * y ** 2) * (y ** 2 - 1) + 2 * (2.625 - x +
                                                                                                        x * y ** 3) * (
                            y ** 3 - 1)
df_y = lambda x, y: 2 * (1.5 - x + x * y) * x + 2 * (2.25 - x + x * y ** 2) * (2 * x * y) + 2 * (2.625 - x +
                                                                                                 x * y ** 3) * (
                            3 * x * y ** 2)
dz_dx = df_x(x, y)
dz_dy = df_y(x, y)
fig, ax = plt.subplots(figsize=(10, 6))
ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
ax.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.5)
ax.plot(*minima_, 'r*', markersize=18)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
x_ = np.vstack((x.reshape(1, -1), y.reshape(1, -1)))
df = lambda x: np.array([
    2 * (1.5 - x[0] + x[0] * x[1]) * (x[1] - 1) + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (x[1] ** 2 - 1) + 2 * (
            2.625 - x[0] + x[0] * x[1] ** 3) * (x[1] ** 3 - 1),
    2 * (1.5 - x[0] + x[0] * x[1]) * x[0] + 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * 2 * x[0] * x[1] + 2 * (
            2.625 - x[0] + x[0] * x[1] ** 3) * 3 * x[0] * x[1] ** 2
])

x0 = np.array([3., 4.])
print("initial point", x0, "gradient", df(x0))
path = alg.gd_adam(df,x0,0.000005,0.9,0.9999,300000,1e-8)
path = np.asarray(path)
print("Extreme point ", path[-1])

plot_path(path, x, y, z, minima_, xmin, xmax, ymin, ymax)
# plt.show()
plt.show()
