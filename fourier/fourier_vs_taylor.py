import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.special import factorial


def taylor_series_exp(x, n_terms):
    """
    Taylor series for e^x centered at x=0:
    e^x = Σ(x^n / n!) for n=0 to n_terms-1
    """
    result = np.zeros_like(x)

    for n in range(n_terms):
        term = (x**n) / factorial(n)
        result += term

    return result


def fourier_series_exp(x, n_terms):
    """
    Fourier series for e^x on the interval [-π, π]:
    e^x ≈ a₀/2 + Σ(aₙcos(nx) + bₙsin(nx))

    Where:
    a₀ = (1/π) ∫[-π,π] e^x dx = (1/π)(e^π - e^(-π))
    aₙ = (1/π) ∫[-π,π] e^x cos(nx) dx = (1/π) * (e^x cos(nx) + n e^x sin(nx))/(1+n²) |[-π,π]
    bₙ = (1/π) ∫[-π,π] e^x sin(nx) dx = (1/π) * (e^x sin(nx) - n e^x cos(nx))/(1+n²) |[-π,π]
    """
    result = np.zeros_like(x)

    # a₀ term
    a0 = (np.exp(np.pi) - np.exp(-np.pi)) / np.pi
    result += a0 / 2

    # Sum over n terms
    for n in range(1, n_terms + 1):
        # Calculate coefficients using the analytical formulas
        cos_term = np.exp(np.pi) * np.cos(n * np.pi) - np.exp(-np.pi) * np.cos(
            -n * np.pi
        )
        sin_term = np.exp(np.pi) * np.sin(n * np.pi) - np.exp(-np.pi) * np.sin(
            -n * np.pi
        )

        a_n = (cos_term + n * sin_term) / (np.pi * (1 + n**2))
        b_n = (sin_term - n * cos_term) / (np.pi * (1 + n**2))

        # Add terms to result
        result += a_n * np.cos(n * x) + b_n * np.sin(n * x)

    return result


# Create x values for plotting
x = np.linspace(-np.pi, np.pi, 1000)

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.25)

# Plot the original e^x function
exp_x = np.exp(x)
(line_original,) = ax.plot(x, exp_x, "b-", linewidth=3, label="e^x")

# Plot Taylor series approximation
(line_taylor,) = ax.plot(
    x, taylor_series_exp(x, 5), "r-", linewidth=2, label="Taylor Series (n=5)"
)

# Plot Fourier series approximation
(line_fourier,) = ax.plot(
    x, fourier_series_exp(x, 5), "g-", linewidth=2, label="Fourier Series (n=5)"
)

# Add horizontal x-axis line at y=0
ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

# Add vertical lines at ±π for clarity
ax.axvline(x=-np.pi, color="gray", linestyle="--", alpha=0.3)
ax.axvline(x=np.pi, color="gray", linestyle="--", alpha=0.3)

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("e^x and its Taylor vs Fourier Series Approximations")
ax.grid(True, alpha=0.3)
ax.legend()

# Set x-axis limits and ticks
ax.set_xlim(-np.pi, np.pi)
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])

# Set y-axis limits
ax.set_ylim(0, 25)

# Create sliders
ax_slider_taylor = plt.axes((0.2, 0.15, 0.6, 0.03))
slider_taylor = Slider(ax_slider_taylor, "Taylor terms", 1, 20, valinit=5, valstep=1)

ax_slider_fourier = plt.axes((0.2, 0.1, 0.6, 0.03))
slider_fourier = Slider(ax_slider_fourier, "Fourier terms", 1, 20, valinit=5, valstep=1)


def update(val):
    n_taylor = int(slider_taylor.val)
    n_fourier = int(slider_fourier.val)

    # Update Taylor series
    taylor_approx = taylor_series_exp(x, n_taylor)
    line_taylor.set_ydata(taylor_approx)
    line_taylor.set_label(f"Taylor Series (n={n_taylor})")

    # Update Fourier series
    fourier_approx = fourier_series_exp(x, n_fourier)
    line_fourier.set_ydata(fourier_approx)
    line_fourier.set_label(f"Fourier Series (n={n_fourier})")

    ax.legend()
    fig.canvas.draw_idle()


slider_taylor.on_changed(update)
slider_fourier.on_changed(update)

plt.show()
