"""
Interactive visualization of Legendre polynomials P_n(x) and Gauss–Legendre nodes.

Features:
- Slider to select degree n (0..20)
- Line mode: only the selected polynomial is visible
- Root mode: hide polynomials and show only Gauss–Legendre nodes (roots of P_n)

Usage:
  python chebyshev/gauss_legendre_visualization.py

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numpy.polynomial.legendre import Legendre, leggauss


# Set to True to enable "root mode": only plot Gauss–Legendre nodes (no lines)
ROOT_MODE = False


def evaluate_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """Evaluate P_n(x) using numpy's Legendre basis for numerical stability."""
    return Legendre.basis(n)(x)


def compute_gauss_legendre_nodes(n: int) -> np.ndarray:
    """Return Gauss–Legendre nodes (roots of P_n) for n>=1; empty array for n=0."""
    if n < 1:
        return np.array([])
    nodes, _weights = leggauss(n)
    return nodes


def main() -> None:
    # Domain for Legendre polynomials
    x_values = np.linspace(-1.0, 1.0, 1200)

    # Prepare figure and axes
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.2)

    # Plot P_0..P_20 and keep their line handles
    max_order = 20
    orders = list(range(max_order + 1))
    cmap = plt.get_cmap("tab10")
    line_handles = []

    for n in orders:
        y = evaluate_legendre(n, x_values)
        (line,) = ax.plot(
            x_values,
            y,
            label=rf"$P_{{{n}}}(x)$",
            color=cmap(n % 10),
            linewidth=2.8 if n == 0 else 1.5,
            alpha=1.0 if n == 0 else 0.9,
            zorder=3 if n == 0 else 2,
            linestyle="-",
        )
        if n != 0:
            line.set_visible(False)
        line_handles.append(line)

    # Prepare a scatter for nodes (updated dynamically)
    nodes_scatter = ax.scatter([], [], marker="o", s=60, color="crimson", zorder=5)

    # Axes styling
    ax.set_title(r"Legendre polynomials $P_n(x)$ for $n=0..20$ (slider highlights $n$)")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$P_n(x)$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    # y = 0 reference line
    ax.axhline(0.0, color="0.2", linewidth=1.0, linestyle="--", alpha=0.6, zorder=1)

    # Slider setup (discrete steps: 0..20)
    slider_ax = plt.axes((0.15, 0.08, 0.7, 0.04))
    order_slider = Slider(
        ax=slider_ax,
        label="Order n",
        valmin=0,
        valmax=max_order,
        valinit=0,
        valstep=1,
    )

    # Initialize visibility depending on ROOT_MODE
    if ROOT_MODE:
        for line in line_handles:
            line.set_visible(False)
        nodes_scatter.set_offsets(np.empty((0, 2)))
        nodes_scatter.set_visible(True)
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_title(
            r"Root mode — Gauss–Legendre nodes of $P_n(x)$ (choose n with slider)"
        )
    else:
        nodes_scatter.set_visible(False)
        for i, line in enumerate(line_handles):
            if i == 0:
                line.set_visible(True)
                line.set_linewidth(2.8)
                line.set_alpha(1.0)
                line.set_zorder(4)
                line.set_linestyle("-")
            else:
                line.set_visible(False)
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.legend(
            handles=[line_handles[0]],
            labels=[rf"$P_{{0}}(x)$"],
            loc="upper center",
            frameon=True,
        )

    def update(selected_n: float) -> None:
        k = int(selected_n)

        if ROOT_MODE:
            for line in line_handles:
                line.set_visible(False)
            nodes = compute_gauss_legendre_nodes(k)
            if nodes.size == 0:
                nodes_scatter.set_offsets(np.empty((0, 2)))
            else:
                xy = np.column_stack((nodes, np.zeros_like(nodes)))
                nodes_scatter.set_offsets(xy)
            nodes_scatter.set_visible(True)
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.set_title(rf"Root mode — Gauss–Legendre nodes of $P_{{{k}}}(x)$")
        else:
            for i, line in enumerate(line_handles):
                if i == k:
                    line.set_visible(True)
                    line.set_linewidth(2.8)
                    line.set_alpha(1.0)
                    line.set_zorder(4)
                    line.set_linestyle("-")
                else:
                    line.set_visible(False)
            nodes_scatter.set_visible(False)
            if ax.legend_ is not None:
                ax.legend_.remove()
            ax.legend(
                handles=[line_handles[k]],
                labels=[rf"$P_{{{k}}}(x)$"],
                loc="upper center",
                frameon=True,
            )
            ax.set_title(
                rf"Legendre polynomials $P_n(x)$ for $n=0..20$ — highlighted: $P_{{{k}}}(x)$"
            )
        ax.figure.canvas.draw_idle()

    order_slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    main()

