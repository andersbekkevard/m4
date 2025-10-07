import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (1 - 3**x) * x**2 + 4 * (x - 1) * 3**x + 4 * (1 - x)


a = -2
b = 3
RESOLUTION = 100
x = np.linspace(a, b, RESOLUTION)
y = f(x)


def plot_function():
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "b-", linewidth=2, label="f(x)")
    plt.plot(x, np.zeros(RESOLUTION), "r--", linewidth=1, label="y = 0")

    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.xlim(a, b)
    plt.xticks(np.arange(a, b + 1, 0.5))

    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    plt.show()


def find_root_bisection_with_convergence(a, b, f, tol=1e-9):
    """
    Bisection method that tracks convergence
    Returns: root, list of midpoints, list of interval widths
    """
    midpoints = []
    interval_widths = []
    iterations = []

    iteration = 0
    while b - a > tol:
        c = (a + b) / 2
        midpoints.append(c)
        interval_widths.append(b - a)
        iterations.append(iteration)

        if f(c) == 0:
            return c, midpoints, interval_widths, iterations
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

        iteration += 1

    final_root = (a + b) / 2
    return final_root, midpoints, interval_widths, iterations


root, midpoints, widths, iterations = find_root_bisection_with_convergence(a, b, f)
print(f"Root found: {root:.10f}")
print(f"Function value at root: {f(root):.2e}")
print(f"Number of iterations: {len(iterations)}")

errors = np.array(widths) / 2

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(iterations, errors, "bo-", linewidth=2, markersize=6)
plt.xlabel("Iteration")
plt.ylabel("Error (log scale)")
plt.title("Bisection Method Convergence")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)
plt.plot(x_plot, y_plot, "b-", linewidth=2, label="f(x)")
plt.axhline(y=0, color="r", linestyle="--", alpha=0.7)
plt.plot(root, f(root), "ro", markersize=8, label=f"Root: {root:.6f}")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function and Root")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
