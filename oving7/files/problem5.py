import numpy as np
import matplotlib.pyplot as plt


def g(y, h):
    return (-1 - h + np.sqrt((1 + h) ** 2 + 4 * h * y)) / (2 * h)


y = 1
h = 0.01
epsilon = 1e-7
max_iterations = 1_000
errors = []
ys = []

for i in range(max_iterations):
    y_old = y
    y = g(y, h)
    error = np.abs(y - y_old)

    ys.append(y)
    errors.append(error)

    if error < epsilon:
        break

print(f"y = {y:.4f}")

plt.plot(np.arange(len(ys)), ys)
plt.xlabel("iteration")
plt.ylabel("y")
plt.grid(True)
plt.show()
