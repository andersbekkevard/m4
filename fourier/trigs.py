import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_trig_functions():
    """
    Plot sin(nx) and cos(nx) from -π to π with interactive slider for n
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25)  # Make room for slider
    
    # Initial parameters
    n_init = 1
    
    # Create x values from -π to π
    x = np.linspace(-np.pi, np.pi, 1000)
    
    # Initial plot
    line_sin, = ax.plot(x, np.sin(n_init * x), 'b-', label=f'sin({n_init}x)', linewidth=2)
    line_cos, = ax.plot(x, np.cos(n_init * x), 'r-', label=f'cos({n_init}x)', linewidth=2)
    
    # Customize the plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trigonometric Functions: sin(nx) and cos(nx)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set x-axis limits and add π markers
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Create slider
    ax_slider = plt.axes((0.2, 0.1, 0.6, 0.03))
    slider = Slider(ax_slider, 'n', 1, 10, valinit=n_init, valstep=1)
    
    def update(val):
        n = slider.val
        line_sin.set_ydata(np.sin(n * x))
        line_cos.set_ydata(np.cos(n * x))
        line_sin.set_label(f'sin({n}x)')
        line_cos.set_label(f'cos({n}x)')
        ax.legend()
        fig.canvas.draw()
    
    slider.on_changed(update)
    
    plt.show()

# Run the interactive plot
if __name__ == "__main__":
    plot_trig_functions()
