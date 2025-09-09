import numpy as np
import matplotlib.pyplot as plt


def p(x):
    return 2 + np.sin(2 * x) / 2


def plot_p():
    x_vals = [0, np.pi / 6, np.pi / 3]
    y_vals = [2, 2 + np.sqrt(3) / 4, 2 + np.sqrt(3) / 4]

    # Also show the original function plot
    x = np.linspace(-np.pi / 4, np.pi / 2, 1000)
    y = p(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="p(x) = 2 + sin(2x)/2")
    plt.scatter(x_vals, y_vals, color="red", s=100, zorder=5, label="Data points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function p(x) with Data Points")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def p_1(x):
    return 2 - 4 * x


def p_2(x):
    return p_1(x) + (11 / 12) * (x + 1) * (x - 1)


x_axis = np.linspace(-3, 7, 100)
plt.plot(x_axis, p_2(x_axis), label="p_2(x)")
points = [[-1, 6], [1, -2], [5, 4]]
plt.scatter(
    [point[0] for point in points],
    [point[1] for point in points],
    color="red",
    s=100,
    zorder=5,
    label="Data points",
)
plt.show()
