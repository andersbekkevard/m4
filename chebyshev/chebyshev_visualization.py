"""
Interactive visualization of the first five Chebyshev polynomials T_n(x), n = 0..4.

Usage:
  python chebyshev/chebyshev_visualization.py

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.polynomial import Chebyshev


# Set to True to enable "root mode": only plot the roots of T_n(x) (no lines)
ROOT_MODE = True


def evaluate_chebyshev_first_kind(n: int, x: np.ndarray) -> np.ndarray:
    """Evaluate T_n(x) using numpy's Chebyshev basis for numerical stability."""
    return Chebyshev.basis(n)(x)


def main() -> None:
    # Domain for Chebyshev polynomials
    x_values = np.linspace(-1.0, 1.0, 1000)

    # Prepare figure and axes
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.2)

    # Plot T_0..T_20 and keep their line handles
    max_order = 20
    orders = list(range(max_order + 1))
    cmap = plt.get_cmap("tab10")
    line_handles = []

    for n in orders:
        y = evaluate_chebyshev_first_kind(n, x_values)
        (line,) = ax.plot(
            x_values,
            y,
            label=rf"$T_{{{n}}}(x)$",
            color=cmap(n),
            linewidth=2.8 if n == 0 else 1.5,
            alpha=1.0 if n == 0 else 0.9,
            zorder=3 if n == 0 else 2,
            linestyle="-",
        )
        # Only show n==0 initially in line mode; root mode will hide all below
        if n != 0:
            line.set_visible(False)
        line_handles.append(line)

    # Prepare a scatter for roots (updated dynamically)
    root_scatter = ax.scatter([], [], marker="o", s=60, color="crimson", zorder=5)

    # Axes styling
    ax.set_title(
        r"Chebyshev polynomials $T_n(x)$ for $n=0..20$ (slider highlights $n$)"
    )
    ax.set_xlabel("x")
    ax.set_ylabel(r"$T_n(x)$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    # y = 0 reference line
    ax.axhline(0.0, color="0.2", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)
    # Legend will be handled dynamically below

    # Slider setup (discrete steps: 0..4)
    slider_ax = plt.axes((0.15, 0.08, 0.7, 0.04))
    order_slider = Slider(
        ax=slider_ax,
        label="Order n",
        valmin=0,
        valmax=max_order,
        valinit=0,
        valstep=1,
    )

    def compute_roots(n: int) -> np.ndarray:
        if n < 1:
            return np.array([])
        k = np.arange(1, n + 1, dtype=float)
        return np.cos((2 * k - 1) * np.pi / (2 * n))

    # Initialize visibility depending on ROOT_MODE
    if ROOT_MODE:
        for line in line_handles:
            line.set_visible(False)
        # Initialize roots for n=0 (empty)
        root_scatter.set_offsets(np.empty((0, 2)))
        root_scatter.set_visible(True)
        # Remove legend if present
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_title(r"Root mode — dots mark roots of $T_n(x)$ (choose n with slider)")
    else:
        root_scatter.set_visible(False)
        # Ensure only T_0 is highlighted initially
        for i, line in enumerate(line_handles):
            if i == 0:
                line.set_visible(True)
                line.set_linewidth(2.8)
                line.set_alpha(1.0)
                line.set_zorder(4)
                line.set_linestyle("-")
            else:
                line.set_visible(False)
        # Create a legend for the active line only
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.legend(
            handles=[line_handles[0]],
            labels=[rf"$T_{{0}}(x)$"],
            loc="upper center",
            frameon=True,
        )

    def update(selected_n: float) -> None:
        # selected_n comes as float from Slider; cast to int
        k = int(selected_n)

        if ROOT_MODE:
            # Only show roots of T_k(x)
            for line in line_handles:
                line.set_visible(False)
            roots = compute_roots(k)
            if roots.size == 0:
                root_scatter.set_offsets(np.empty((0, 2)))
            else:
                # Build Nx2 array of (x, 0)
                xy = np.column_stack((roots, np.zeros_like(roots)))
                root_scatter.set_offsets(xy)
            root_scatter.set_visible(True)
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.set_title(rf"Root mode — roots of $T_{{{k}}}(x)$")
        else:
            # Update line styles to highlight the selected order
            for i, line in enumerate(line_handles):
                if i == k:
                    line.set_visible(True)
                    line.set_linewidth(2.8)
                    line.set_alpha(1.0)
                    line.set_zorder(4)
                    line.set_linestyle("-")
                else:
                    line.set_visible(False)
            root_scatter.set_visible(False)
            # Update legend to only show active polynomial
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.legend(
                handles=[line_handles[k]],
                labels=[rf"$T_{{{k}}}(x)$"],
                loc="upper center",
                frameon=True,
            )
            ax.set_title(
                rf"Chebyshev polynomials $T_n(x)$ for $n=0..20$ — highlighted: $T_{{{k}}}(x)$"
            )
        ax.figure.canvas.draw_idle()

    order_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
