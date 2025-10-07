"""
Interactive Newton's Method visualizer

How to use:
- Edit the function f(x) and the initial guess x0 below.
- Run this script. You'll see f(x) plotted along with the point at x0.
- Use the slider to step through Newton iterations. Each step shows the
  tangent at the previous iterate and the new x-intercept (the next iterate).
"""

from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# -----------------------------------
# 1) Define your function and x0 here
# -----------------------------------
def f(x: np.ndarray | float) -> np.ndarray | float:
    """Function to find a root of.

    Modify this function to explore different behaviors.
    Example below: f(x) = x^3 - x - 2 which has a real root near ~1.521.
    """
    return x**3 - x - 2


# Initial guess
x0: float = 1.5

# Maximum number of iterations available on the slider
max_iterations: int = 12

# Stopping tolerance for precomputing the iteration sequence
tol: float = 1e-12


# ------------------------------
# Helper: numerical derivative
# ------------------------------
def numerical_derivative(
    func: Callable[[np.ndarray | float], np.ndarray | float], x: float
) -> float:
    """Compute f'(x) using a central difference with an adaptive step.

    This keeps the code simple so you can define f(x) as plain Python.
    """
    h = 1e-6 * (1.0 + abs(x))
    return float((func(x + h) - func(x - h)) / (2.0 * h))


# ---------------------------------
# Precompute Newton's method steps
# ---------------------------------
def newton_sequence(
    func: Callable[[np.ndarray | float], np.ndarray | float],
    x_start: float,
    max_iter: int,
    tolerance: float,
) -> List[float]:
    """Return the sequence [x0, x1, ..., xN] produced by Newton's method.

    The length will be at most max_iter + 1 and may be shorter if we
    encounter a near-zero derivative or converge early.
    """
    xs: List[float] = [x_start]
    for _ in range(max_iter):
        xk = xs[-1]
        yk = float(func(xk))
        dyk = numerical_derivative(func, xk)
        # Guard against division by zero or non-finite derivative
        if not np.isfinite(dyk) or abs(dyk) < 1e-14:
            break
        x_next = xk - yk / dyk
        xs.append(float(x_next))
        if abs(x_next - xk) < tolerance:
            break
    return xs


xs_all = newton_sequence(f, x0, max_iterations, tol)


# -------------------
# Build the plot area
# -------------------
def compute_domain(points: List[float]) -> tuple[float, float]:
    """Choose an x-domain that comfortably contains the iteration points."""
    xmin = min(points)
    xmax = max(points)
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    span = xmax - xmin
    pad = 0.35 * span + 1.0  # add breathing room
    return xmin - pad, xmax + pad


x_min, x_max = compute_domain(xs_all)
xx = np.linspace(x_min, x_max, 1000)

fig, ax = plt.subplots(figsize=(9, 5))
plt.subplots_adjust(bottom=0.25)

# Plot f(x)
(func_line,) = ax.plot(xx, f(xx), color="tab:blue", lw=2.0, label="f(x)")
ax.axhline(0.0, color="black", lw=1.0)

# Initial point (x0, f(x0))
(point_line,) = ax.plot(
    [x0], [float(f(x0))], marker="o", color="tab:red", ms=8, label="x_n on curve"
)

# Tangent line at x_{n-1} (hidden initially)
(tangent_line,) = ax.plot(
    [np.nan], [np.nan], color="tab:orange", lw=1.75, label="tangent at x_{n-1}"
)

# Marker for the new root x_n on the x-axis (hidden initially)
(root_marker,) = ax.plot(
    [np.nan],
    [np.nan],
    marker="x",
    color="tab:green",
    ms=9,
    linestyle="none",
    label="x_n (new iterate)",
)

ax.set_xlim(x_min, x_max)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Newton's Method Visualizer")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")


# -----------------------------
# Slider: iteration 0..N
#  - 0 shows only f(x) and x0
#  - n>=1 shows tangent at x_{n-1} and the new root x_n
# -----------------------------
slider_ax = plt.axes((0.15, 0.1, 0.7, 0.04))
iter_slider = Slider(
    ax=slider_ax,
    label="Iterations",
    valmin=0,
    valmax=len(xs_all) - 1,
    valinit=0,
    valstep=1,
)


def update(_value: float) -> None:
    n = int(iter_slider.val)

    # Always show the current point on the curve at x_n
    x_n = xs_all[n]
    y_n = float(f(x_n))
    point_line.set_data([x_n], [y_n])

    if n == 0:
        # Hide tangent and root marker on initial view
        tangent_line.set_data([np.nan], [np.nan])
        root_marker.set_data([np.nan], [np.nan])
    else:
        # Show tangent at x_{n-1}
        x_prev = xs_all[n - 1]
        y_prev = float(f(x_prev))
        dy_prev = numerical_derivative(f, x_prev)

        # Compute tangent line y = y_prev + dy_prev * (t - x_prev) across current x-limits
        cur_xlim = ax.get_xlim()
        t = np.linspace(cur_xlim[0], cur_xlim[1], 2)
        tangent_line.set_data(t, y_prev + dy_prev * (t - x_prev))

        # The new iterate is the x-intercept of that tangent line (already xs_all[n])
        root_marker.set_data([x_n], [0.0])

        # Expand x-limits if the new point falls outside the view
        xmin, xmax = cur_xlim
        if x_n < xmin or x_n > xmax:
            margin = 0.15 * (xmax - xmin)
            new_xmin = min(xmin, x_n) - margin
            new_xmax = max(xmax, x_n) + margin
            ax.set_xlim(new_xmin, new_xmax)
            # Recompute function line across the new domain
            new_xx = np.linspace(new_xmin, new_xmax, 1000)
            func_line.set_data(new_xx, f(new_xx))

    fig.canvas.draw_idle()


# Initialize view
update(0)
iter_slider.on_changed(update)
plt.tight_layout()
plt.show()
