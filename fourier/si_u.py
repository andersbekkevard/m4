import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Set modern matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")

# Set font to support Unicode characters
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# Configuration constants
PLOT_CONFIG = {
    "figsize": (14, 8),
    "dpi": 100,
    "x_range": (-10, 10),
    "y_range": (-1.8, 2),
}

COLOR_SCHEME = {
    "si": "#1f77b4",  # Deep blue
    "integrand": "#ff7f0e",  # Deep orange
    "accent": "#d62728",  # Red
}

STYLE_CONFIG = {
    "line_width": 2.5,
    "grid_alpha": 0.2,
    "grid_style": "--",
    "font_size_label": 13,
    "font_size_title": 15,
    "font_size_tick": 11,
    "font_size_annotation": 10,
}


def Si(u):
    """
    Sine integral function: Si(u) = ∫₀ᵘ sin(x)/x dx

    Args:
        u: Upper limit of integration

    Returns:
        The sine integral value at u
    """

    def integrand(x):
        """Integrand sin(x)/x with singularity handling at x=0"""
        if x == 0:
            return 1.0
        return np.sin(x) / x

    result, _ = quad(integrand, 0, u)
    return result


def _calculate_si_values(u_values):
    """Calculate Si(u) for all u values"""
    return [Si(u) for u in u_values]


def _calculate_integrand_values(u_values):
    """Calculate sin(u)/u for all u values with singularity handling"""
    integrand_values = []
    for u in u_values:
        if u == 0:
            integrand_values.append(1.0)  # lim_{u→0} sin(u)/u = 1
        else:
            integrand_values.append(np.sin(u) / u)
    return integrand_values


def _plot_curves(ax, u_values, si_values, integrand_values):
    """Plot the Si(u) and sin(u)/u curves"""
    ax.plot(
        u_values,
        si_values,
        color=COLOR_SCHEME["si"],
        linewidth=STYLE_CONFIG["line_width"],
        label="Si(u) = ∫₀ᵘ sin(x)/x dx",
        zorder=3,
    )
    ax.plot(
        u_values,
        integrand_values,
        color=COLOR_SCHEME["integrand"],
        linewidth=STYLE_CONFIG["line_width"],
        label="sin(u)/u",
        alpha=0.85,
        zorder=2,
    )


def _configure_grid_and_axes(ax):
    """Configure grid and reference lines"""
    ax.grid(
        True,
        alpha=STYLE_CONFIG["grid_alpha"],
        linestyle=STYLE_CONFIG["grid_style"],
        linewidth=0.7,
    )

    # Subtle reference lines
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3, zorder=1)
    ax.axhline(
        y=np.pi / 2,
        color=COLOR_SCHEME["accent"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="y = π/2 (asymptote)",
        zorder=1,
    )
    ax.axhline(
        y=-np.pi / 2,
        color=COLOR_SCHEME["accent"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="y = -π/2 (asymptote)",
        zorder=1,
    )


def _configure_labels_and_ticks(ax):
    """Configure axis labels, title, and tick marks"""
    ax.set_xlabel("u", fontsize=STYLE_CONFIG["font_size_label"], fontweight="bold")
    ax.set_ylabel("Value", fontsize=STYLE_CONFIG["font_size_label"], fontweight="bold")
    ax.set_title(
        "Sine Integral Function and Its Integrand",
        fontsize=STYLE_CONFIG["font_size_title"],
        fontweight="bold",
        pad=20,
    )

    # Set x-axis ticks to multiples of π
    x_ticks = np.arange(-3 * np.pi, 4 * np.pi, np.pi)
    x_labels = [f"{int(tick / np.pi)}π" if tick != 0 else "0" for tick in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=STYLE_CONFIG["font_size_tick"])
    ax.tick_params(axis="y", labelsize=STYLE_CONFIG["font_size_tick"])


def _mark_key_points(ax):
    """Mark and annotate key points on the curve"""
    key_points = [
        (0, 0, "Si(0) = 0"),
        (np.pi / 2, Si(np.pi / 2), f"Si(π/2) ≈ {Si(np.pi / 2):.3f}"),
        (np.pi, Si(np.pi), f"Si(π) ≈ {Si(np.pi):.3f}"),
    ]

    for x_val, y_val, label in key_points:
        ax.plot(
            x_val,
            y_val,
            "o",
            color=COLOR_SCHEME["accent"],
            markersize=7,
            zorder=4,
            markeredgewidth=1.5,
            markeredgecolor="darkred",
        )
        ax.annotate(
            label,
            xy=(x_val, y_val),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=STYLE_CONFIG["font_size_annotation"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8,
            ),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0", color="gray", lw=0.8
            ),
        )


def plot_sine_integral():
    """Generate and display a comprehensive plot of the sine integral function"""
    # Calculate values
    u_values = np.linspace(PLOT_CONFIG["x_range"][0], PLOT_CONFIG["x_range"][1], 1000)
    si_values = _calculate_si_values(u_values)
    integrand_values = _calculate_integrand_values(u_values)

    # Create figure
    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["figsize"],
        dpi=PLOT_CONFIG["dpi"],
    )

    # Plot components
    _plot_curves(ax, u_values, si_values, integrand_values)
    _configure_grid_and_axes(ax)
    _configure_labels_and_ticks(ax)
    _mark_key_points(ax)

    # Finalize plot
    ax.legend(
        loc="lower right",
        fontsize=STYLE_CONFIG["font_size_label"],
        framealpha=0.95,
        edgecolor="gray",
    )
    ax.set_xlim(PLOT_CONFIG["x_range"])
    ax.set_ylim(PLOT_CONFIG["y_range"])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sine_integral()
