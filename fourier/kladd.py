import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

a0 = 1 / pi
an = lambda n: 2 * np.sin(n) / (n * pi)

x = np.linspace(-pi, pi, 1000)
n_values = np.arange(1, 10000)
y = a0 + np.sum(
    an(n_values)[:, np.newaxis] * np.cos(n_values[:, np.newaxis] * x), axis=0
)

plt.plot(x, y)
plt.show()
