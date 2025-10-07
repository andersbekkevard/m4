"""Interactive Newton's method visualisation using matplotlib sliders.

Adjust `f(x)` and `x0` below to explore how Newton's method converges.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

a = 1.0


def f(x: np.ndarray | float) -> np.ndarray | float:
    """Function whose root we want to approximate.

    Implements: (x³ - 3a²x - 1)² + (3ax² - a³)²
    """
    global a
    term1 = x**3 - 3 * a**2 * x - 1
    term2 = 3 * a * x**2 - a**3
    return term1**2 + term2**2


# Initial guess for the root
x0 = 25


def _derivative(func, x: float, h: float = 1e-5) -> float:
    """Return the symmetric finite-difference derivative of `func` at `x`."""
    return (func(x + h) - func(x - h)) / (2 * h)


def _prepare_iterations(max_iter: int = 15) -> tuple[np.ndarray, list[float]]:
    """Compute Newton iterations and corresponding derivatives up to `max_iter`."""
    iterates = [x0]
    derivatives: list[float] = []

    for _ in range(max_iter):
        x_n = iterates[-1]
        derivative = _derivative(f, x_n)
        if np.isclose(derivative, 0.0):
            break
        x_next = x_n - f(x_n) / derivative
        derivatives.append(derivative)
        iterates.append(float(x_next))

    return np.array(iterates), derivatives


def visualise_newtons_method(max_iter: int = 15, padding: float = 1.0) -> None:
    """Render an interactive plot that exposes Newton iterations through a slider."""
    iterates, derivatives = _prepare_iterations(max_iter=max_iter)

    if iterates.size == 0:
        raise ValueError(
            "No iterations produced. Check that `x0` and `f(x)` are defined correctly."
        )

    slider_max = len(derivatives)

    # Protect against degenerate x ranges by padding with a sensible default span.
    x_min = float(np.min(iterates)) - padding
    x_max = float(np.max(iterates)) + padding
    if np.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0

    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = f(x_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.22)

    ax.set_title("Newton's Method Visualiser")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    ax.plot(x_vals, y_vals, label="f(x)")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.scatter([x0], [float(f(x0))], color="orange", zorder=5, label="$x_0$")

    # Artists updated by the slider callback.
    (current_point,) = ax.plot([], [], "o", color="tab:blue", label="Iteration point")
    (root_marker,) = ax.plot(
        [], [], "x", markersize=8, color="tab:red", label="Next root estimate"
    )
    (tangent_line,) = ax.plot(
        [], [], color="tab:green", linewidth=1.2, label="Tangent line"
    )
    (vertical_line,) = ax.plot(
        [],
        [],
        "--",
        color="tab:orange",
        linewidth=1.0,
        label="Function value at estimate",
    )

    iteration_info = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )

    ax.legend(loc="upper right")

    slider_ax = fig.add_axes((0.15, 0.08, 0.7, 0.04))
    iteration_slider = Slider(
        slider_ax,
        label="Iterations",
        valmin=0,
        valmax=slider_max,
        valinit=0,
        valstep=1,
    )

    def update(_):
        iteration = int(iteration_slider.val)

        if iteration == 0:
            current_point.set_data([x0], [float(f(x0))])
            root_marker.set_data([], [])
            tangent_line.set_data([], [])
            tangent_line.set_visible(False)
            vertical_line.set_data([], [])
            vertical_line.set_visible(False)
            message = f"x_0 = {iterates[0]:.6f}"
            if slider_max == 0:
                message += "\nDerivative is ~0 at x_0; adjust the guess to iterate."
            iteration_info.set_text(message)
        else:
            base_index = iteration - 1
            x_base = iterates[base_index]
            y_base = float(f(x_base))
            derivative = derivatives[base_index]

            tangent_line.set_visible(True)
            tangent_line.set_data(x_vals, derivative * (x_vals - x_base) + y_base)

            current_point.set_data([x_base], [y_base])
            root_marker.set_data([iterates[iteration]], [0.0])

            # Draw vertical line from root estimate to function value
            vertical_line.set_visible(True)
            root_x = iterates[iteration]
            root_y = float(f(root_x))
            vertical_line.set_data([root_x, root_x], [0.0, root_y])

            iteration_info.set_text(
                f"x_{base_index} = {x_base:.6f}\n"
                f"x_{iteration} = {iterates[iteration]:.6f}"
            )

        fig.canvas.draw_idle()

    # Initialise the artists.
    update(0)

    iteration_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    visualise_newtons_method()
