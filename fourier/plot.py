import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


def plot_fourier_series(f, N, L, a0=0.0, an=lambda n: 0.0, bn=lambda n: 0.0):
    """
    Plot the Fourier series of a function f

    Parameters:
    -----------
    f : callable
        The original function to approximate
    N : int, array-like, or tuple
        Number of terms in the Fourier series. Can be a single int or an iterable of ints
    L : float
        Half-period of the function
    a0 : float, optional
        The constant (zeroth) Fourier coefficient (default is 0)
    an : callable, optional
        Function that returns the nth cosine coefficient
    bn : callable, optional
        Function that returns the nth sine coefficient
    """
    
    # Convert N to a list if it's a single int
    if isinstance(N, int):
        N_values = [N]
    else:
        N_values = list(N)
    
    x = np.linspace(-L, L, 1000)
    
    # Create figure and explicit axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot original function first
    ax.plot(x, f(x), "r--", linewidth=2.5, label="Original Function", zorder=10)
    
    # Generate distinct colors for each N value with good contrast
    # Colors ordered to maximize contrast for any subset (1-10)
    colors = [
        (0.000, 0.450, 0.700),  # Blue
        (0.000, 0.620, 0.450),  # Teal
        (0.835, 0.370, 0.000),  # Orange
        (0.910, 0.820, 0.000),  # Yellow
        (0.620, 0.000, 0.450),  # Purple
        (0.340, 0.710, 0.910),  # Sky Blue
        (0.870, 0.560, 0.000),  # Amber
        (0.000, 0.730, 0.830),  # Cyan
        (0.500, 0.500, 0.500),  # Gray
        (0.000, 0.340, 0.170),  # Forest Green
    ]
    colors = colors[:len(N_values)]
    
    # Plot each Fourier series with different N
    for idx, n_val in enumerate(N_values):
        y = np.zeros_like(x)
        y += a0
        for n in range(1, n_val + 1):
            y += an(n) * np.cos(n * pi * x / L) + bn(n) * np.sin(n * pi * x / L)
        ax.plot(x, y, color=colors[idx], linewidth=2, label=f"Fourier Series (N={n_val})")
    
    # Add labels and title
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title("Fourier Series Approximation", fontsize=14, fontweight="bold")
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--")
    
    # Add x and y axes through the origin (only if they intersect the plot area)
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    
    # Only draw axes if they intersect the visible plot area
    if y_min <= 0 <= y_max:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.6, zorder=0)
    if x_min <= 0 <= x_max:
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.6, zorder=0)
    
    # Add legend
    ax.legend(fontsize=11, loc="best")
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.show()

def plot_old_series():
    # plot_fourier_series(
    #     lambda x: (1 - x),
    #     N=50,
    #     L=1,
    #     an=lambda n: 0,
    #     bn=lambda n: 2 * (-1) ** n / (n * pi),
    # )
    # plot_fourier_series(
    #     lambda x: np.abs(np.cos(3 * pi * x)),
    #     N=50,
    #     L=2 / pi,
    #     an=lambda n: 1 / pi * (
    #         (2 * (-1) ** n / (2 * n - 1)) +
    #         (2 * (-1) ** n / (2 * n + 1))
    #     ),
    #     bn=lambda n: 0,
    # )
    # plot_fourier_series(
    #     lambda x: np.abs(np.sin(x)),
    #     N=50,
    #     L=pi / 2,
    #     a0=2 / pi,
    #     an=lambda n: 2 / pi * (1 / (2 * n + 1) - 1 / (2 * n - 1)),
    #     bn=lambda n: 0,
    # )
    # plot_fourier_series(
    #     lambda x: (x**2 - x),
    #     N=100,
    #     L=1,
    #     a0=1 / 3,
    #     an=lambda n: 4 * (-1) ** n / (n**2 * pi**2),
    #     bn=lambda n: 2 * (-1) ** n / (n * pi),
    # )

    # plot_fourier_series(
    #     lambda x: (np.sin(x) ** 2 + 3 * x ** 2 - 4 * x + 5),
    #     N=300,
    #     L=pi,
    #     a0=pi**2 + 11 / 2,
    #     an=lambda n: 12 * (-1) ** n / n**2 + (-1/2 if n == 2 else 0), # needs a delta term for n = 2
    #     bn=lambda n: 8 * (-1) ** n / n,
    # )
    return

if __name__ == "__main__":
    # Problem 4a)
    plot_fourier_series(
        lambda x: np.where((x >= 0) & (x <= pi/2), x, 0),
        N=[5, 20, 100],
        L=pi,
        a0=pi/16,
        an=lambda n: (pi*n*np.sin(n*pi/2)/2 + np.cos(n*pi/2) - 1) / (pi*n**2),
        bn=lambda n: -np.cos(n*pi/2)/(2*n) + np.sin(n*pi/2)/(pi*n**2),
    )
    
    # Problem 4b)
