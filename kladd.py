import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_fourier_series():
    """
    Plot Fourier series approximation of a square wave with interactive k-slider
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)  # Make room for slider
    
    # Initial parameters
    k_init = 10
    x = np.linspace(-np.pi, np.pi, 1000)
    
    # Calculate initial Fourier series
    y = np.zeros_like(x)
    for i in range(1, k_init + 1):
        y += 2/(np.pi * i) * (1 - (-1)**i) * np.sin(i*x)
    
    # Initial plot
    line, = ax.plot(x, y, 'b-', linewidth=2, label=f'Fourier series (k={k_init})')
    
    # Plot theoretical square wave for comparison
    square_wave = np.sign(x)
    ax.plot(x, square_wave, 'r--', linewidth=1, alpha=0.7, label='Square wave')
    
    # Customize the plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Fourier Series Approximation of Square Wave')
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
    slider = Slider(ax_slider, 'k', 1, 50, valinit=k_init, valstep=1)
    
    def update(val):
        k = int(slider.val)
        y_new = np.zeros_like(x)
        for i in range(1, k + 1):
            y_new += 2/(np.pi * i) * (1 - (-1)**i) * np.sin(i*x)
        
        line.set_ydata(y_new)
        line.set_label(f'Fourier series (k={k})')
        ax.legend()
        fig.canvas.draw()
    
    slider.on_changed(update)
    
    plt.show()

# Run the interactive plot
if __name__ == "__main__":
    plot_fourier_series()