import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.polynomial import chebyshev as C


def rectangular_wave(x, k=1):
    """
    Rectangular wave function:
    f(x) = -k for -π < x < 0
    f(x) = k for 0 < x < π
    Periodic with period 2π
    """
    # Handle the discontinuity at x = 0 by using np.where
    result = np.where(x < 0, -k, k)

    # Handle the case where x is exactly 0 (or multiples of 2π)
    # For plotting purposes, we'll set it to 0
    result = np.where(np.abs(x % (2 * np.pi)) < 1e-10, 0, result)

    return result


def fourier_series(x, n_terms, k=1):
    """
    Fourier series for the rectangular wave:
    f(x) = Σ(b_n * sin(nx)) where b_n = 4k/(nπ) for odd n, 0 for even n
    """
    result = np.zeros_like(x)

    for n in range(1, n_terms + 1):
        if n % 2 == 1:  # Only odd terms contribute
            b_n = 4 * k / (n * np.pi)
            result += b_n * np.sin(n * x)

    return result


def best_polynomial_approx(x, degree, k=1):
    """
    Compute the best polynomial approximation using Chebyshev interpolation.
    This finds the polynomial of given degree that minimizes max error.
    """
    # Sample points for Chebyshev interpolation (in [-pi, pi])
    n_sample = max(degree + 1, 200)
    x_cheb = np.cos(
        np.pi * np.arange(n_sample) / (n_sample - 1)
    )  # Chebyshev nodes in [-1, 1]
    x_cheb = x_cheb * np.pi  # Scale to [-π, π]
    y_cheb = rectangular_wave(x_cheb, k)

    # Fit Chebyshev polynomial
    cheb_coeffs = C.chebfit(x_cheb, y_cheb, degree)

    # Evaluate polynomial at x points
    # Map x from [-π, π] to [-1, 1] for Chebyshev evaluation
    x_mapped = x / np.pi
    poly_vals = C.chebval(x_mapped, cheb_coeffs)

    return poly_vals


# Create x values for plotting
x = np.linspace(-np.pi, np.pi, 1000)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.25)

# Plot the original rectangular wave
(line_original,) = ax.plot(
    x, rectangular_wave(x, k=1), "b-", linewidth=2, label="Original f(x)"
)
(line_fourier,) = ax.plot(
    x, fourier_series(x, 1, k=1), "r-", linewidth=2, label="Fourier Series (n=1)"
)

# Plot the polynomial approximation
(line_poly,) = ax.plot(
    x,
    best_polynomial_approx(x, 5, k=1),
    "g--",
    linewidth=2,
    label="Polynomial (degree=5)",
)

# Add horizontal x-axis line at y=0
ax.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=1)

# Add vertical lines at discontinuities for clarity
for n in range(-2, 3):
    ax.axvline(x=n * 2 * np.pi, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=n * 2 * np.pi + np.pi, color="gray", linestyle="--", alpha=0.3)

ax.set_xlabel("x (units of π)")
ax.set_ylabel("f(x)")
ax.set_title("Rectangular Wave Function and Best Polynomial Approximation")
ax.grid(True, alpha=0.3)
ax.legend()

# Set x-axis ticks in units of π
ax.set_xticks(np.arange(-1, 2) * np.pi)
ax.set_xticklabels([f"{i}π" for i in range(-1, 2)])
ax.set_xlim(-np.pi, np.pi)

ax.set_ylim(-1.5, 1.5)

# Create sliders
ax_slider_poly = plt.axes((0.2, 0.15, 0.6, 0.03))
slider_poly = Slider(ax_slider_poly, "Polynomial degree", 1, 100, valinit=5, valstep=1)

ax_slider_fourier = plt.axes((0.2, 0.1, 0.6, 0.03))
slider_fourier = Slider(
    ax_slider_fourier, "Fourier terms", 1, 100, valinit=1, valstep=1
)


def update(val):
    degree = int(slider_poly.val)
    n_terms = int(slider_fourier.val)

    poly_approx = best_polynomial_approx(x, degree, k=1)
    line_poly.set_ydata(poly_approx)
    line_poly.set_label(f"Polynomial (degree={degree})")

    fourier_approx = fourier_series(x, n_terms, k=1)
    line_fourier.set_ydata(fourier_approx)
    line_fourier.set_label(f"Fourier Series (n={n_terms})")

    ax.legend()
    fig.canvas.draw_idle()


slider_poly.on_changed(update)
slider_fourier.on_changed(update)

plt.show()
