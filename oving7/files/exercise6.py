import numpy as np
import matplotlib.pyplot as plt


def f(t, y):
    return -2 * t * y


def dfdt(t, y):
    return -2 * y


def dfdy(t, y):
    return -2 * t


def df(t, y):
    return dfdt(t, y) + dfdy(t, y) * f(t, y)


t = 0
y = 1
h = 0.1


def exact_solution(t):
    return np.exp(-(t**2))


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


compare_solutions(
    0,
    1,
    0.2,
    2.0,
    eulers_method,
    heuns_method,
    taylor_2_method,
)
