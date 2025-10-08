import numpy as np
import matplotlib.pyplot as plt


# Definitions for the ODE
def f(t, y):
    return -2 * t * y


def dfdt(t, y):
    return -2 * y


def dfdy(t, y):
    return -2 * t


def df(t, y):
    return dfdt(t, y) + dfdy(t, y) * f(t, y)


def exact_solution(t):
    return np.exp(-(t**2))


# Methods
def eulers_method(t_0, y_0, h, t_target):
    t = t_0
    y = y_0
    result = [(t, y)]
    while t < t_target:
        y = y + h * f(t, y)
        t = t + h
        result.append((t, y))
    return result


def heuns_method(t_0, y_0, h, t_target):
    t = t_0
    y = y_0
    result = [(t, y)]
    while t < t_target:
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        y = y + h / 2 * (k1 + k2)
        t = t + h
        result.append((t, y))
    return result


def taylor_2_method(t_0, y_0, h, t_target):
    t = t_0
    y = y_0
    result = [(t, y)]
    while t < t_target:
        y = y + h * f(t, y) + (h**2 / 2) * df(t, y)
        t = t + h
        result.append((t, y))
    return result


# Vectorized methods
def eulers_method_vectorized(t_array, y_0):
    """
    Vectorized Euler's method that takes an array of t values and returns array of y values.

    Args:
        t_array: Array of time points [t_0, t_1, t_2, ..., t_n]
        y_0: Initial value y(t_0)

    Returns:
        Array of y values corresponding to each t in t_array
    """
    y = np.zeros(len(t_array))
    y[0] = y_0

    for i in range(len(t_array) - 1):
        h = t_array[i + 1] - t_array[i]
        y[i + 1] = y[i] + h * f(t_array[i], y[i])

    return y


def heuns_method_vectorized(t_array, y_0):
    """
    Vectorized Heun's method that takes an array of t values and returns array of y values.

    Args:
        t_array: Array of time points [t_0, t_1, t_2, ..., t_n]
        y_0: Initial value y(t_0)

    Returns:
        Array of y values corresponding to each t in t_array
    """
    y = np.zeros(len(t_array))
    y[0] = y_0

    for i in range(len(t_array) - 1):
        h = t_array[i + 1] - t_array[i]
        k1 = f(t_array[i], y[i])
        k2 = f(t_array[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h / 2 * (k1 + k2)

    return y


def taylor_2_method_vectorized(t_array, y_0):
    """
    Vectorized Taylor's 2nd order method that takes an array of t values and returns array of y values.

    Args:
        t_array: Array of time points [t_0, t_1, t_2, ..., t_n]
        y_0: Initial value y(t_0)

    Returns:
        Array of y values corresponding to each t in t_array
    """
    y = np.zeros(len(t_array))
    y[0] = y_0

    for i in range(len(t_array) - 1):
        h = t_array[i + 1] - t_array[i]
        y[i + 1] = y[i] + h * f(t_array[i], y[i]) + (h**2 / 2) * df(t_array[i], y[i])

    return y


# Comparison plotting
def compare_solutions(t_0, y_0, h, t_target, *functions):
    plt.figure(figsize=(12, 8))

    # Plot each numerical method
    colors = ["blue", "green", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "D", "v"]

    for i, function in enumerate(functions):
        result = function(t_0, y_0, h, t_target)
        t = [point[0] for point in result]
        y = [point[1] for point in result]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(
            t,
            y,
            color=color,
            marker=marker,
            linestyle="-",
            label=f"{function.__name__} (h={h})",
            markersize=4,
            linewidth=1.5,
            zorder=2
            + i,  # Higher zorder = on top. Numerical methods get zorder 2, 3, 4, ...
        )

    # Plot exact solution
    t_exact = np.linspace(t_0, t_target, 1000)
    plt.plot(
        t_exact,
        exact_solution(t_exact),
        "r-",
        label="Exact Solution",
        linewidth=4,
        zorder=1,  # Lower zorder = underneath numerical methods
    )

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Comparison: Numerical Methods vs Exact Solution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def compare_solutions_vectorized(t_array, y_0, *functions):
    """
    Compare vectorized methods that take a t_array and return y_array.

    Args:
        t_array: Array of time points to evaluate at
        y_0: Initial value y(t_0)
        *functions: Vectorized method functions to compare
    """
    plt.figure(figsize=(12, 8))

    # Plot each numerical method
    colors = ["blue", "green", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "D", "v"]

    for i, function in enumerate(functions):
        y = function(t_array, y_0)

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(
            t_array,
            y,
            color=color,
            marker=marker,
            linestyle="-",
            label=f"{function.__name__}",
            markersize=4,
            linewidth=1.5,
            zorder=2 + i,
        )

    # Plot exact solution
    t_exact = np.linspace(t_array[0], t_array[-1], 1000)
    plt.plot(
        t_exact,
        exact_solution(t_exact),
        "r-",
        label="Exact Solution",
        linewidth=4,
        zorder=1,
    )

    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Comparison: Vectorized Numerical Methods vs Exact Solution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Convergence
def approximate_convergence_order(h_1=1e-2, h_2=1e-3, y_0=1.0):
    t_1 = np.arange(0, 2, h_1)
    t_2 = np.arange(0, 2, h_2)

    error_1 = np.max(np.abs(eulers_method_vectorized(t_1, y_0) - exact_solution(t_1)))
    error_2 = np.max(np.abs(eulers_method_vectorized(t_2, y_0) - exact_solution(t_2)))
    p = np.log(error_1 / error_2) / np.log(h_1 / h_2)
    print(f"p for Euler's method: {p}")

    error_1 = np.max(np.abs(heuns_method_vectorized(t_1, y_0) - exact_solution(t_1)))
    error_2 = np.max(np.abs(heuns_method_vectorized(t_2, y_0) - exact_solution(t_2)))
    p = np.log(error_1 / error_2) / np.log(h_1 / h_2)
    print(f"p for Heun's method: {p}")

    error_1 = np.max(np.abs(taylor_2_method_vectorized(t_1, y_0) - exact_solution(t_1)))
    error_2 = np.max(np.abs(taylor_2_method_vectorized(t_2, y_0) - exact_solution(t_2)))
    p = np.log(error_1 / error_2) / np.log(h_1 / h_2)
    print(f"p for Taylor's 2nd order method: {p}")

    """
    Output:
    p for Euler's method: 1.0028646196449251
    p for Heun's method: 2.0046147808408326
    p for Taylor's 2nd order method: 2.0014391055154577
    """
