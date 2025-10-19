import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


def integrate(f, a, b, n=1000):
    """Calculate the integral of function f from a to b using trapezoidal rule."""
    x = np.linspace(a, b, n)
    return np.trapezoid(f(x), x)


def main():
    """Plot and calculate areas under |sin(nx)| curves."""
    # Integration bounds
    a, b = 0, pi

    # Setup plot
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 4))

    # Calculate and plot for n = 1 to 4
    for n in range(1, 5):
        # Define function
        func = lambda x: np.abs(np.sin(n * x))

        # Calculate integral
        integral_value = integrate(func, a, b)
        print(f"∫₀^π |sin({n}x)| dx = {integral_value:.3f}")

        # Plotting data
        x_plot = np.linspace(a, b, 1000)
        y_plot = func(x_plot)

        # Plot curve and fill area
        plt.plot(
            x_plot,
            y_plot,
            color=colors[n - 1],
            linewidth=2,
            label=f"|sin({n}x)|, Area = {integral_value:.3f}",
        )
        plt.fill_between(x_plot, y_plot, alpha=0.3, color=colors[n - 1])

    # Formatting
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Area under |sin(nx)| curves from 0 to π")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
