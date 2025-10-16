"""
Fourier Series Symbolic Computation and Visualization
Interactive tool for computing and plotting Fourier series
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sympy as sp
from sympy import symbols, pi, sin, cos, integrate, simplify, latex

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Define the symbolic variable
x = symbols('x', real=True)

# Define your function f(x)
# Examples:
# f = x**2
# f = sp.Abs(x)
# f = sp.Piecewise((1, x > 0), (-1, True))
# f = sp.exp(x)
f = 1-x

# Define L (period is from -L to L)
L = 1

# Number of terms to compute symbolically (adjust based on complexity)
N_TERMS = 10

# ============================================================================
# SYMBOLIC FOURIER COEFFICIENT COMPUTATION
# ============================================================================

print("=" * 70)
print("FOURIER SERIES COMPUTATION")
print("=" * 70)
print(f"\nFunction: f(x) = {f}")
print(f"Period: [{-L}, {L}]")
print(f"Computing {N_TERMS} terms...\n")

# Compute a0 (constant term)
print("Computing a₀...")
a0_integral = integrate(f, (x, -L, L)) / (2*L)
a0 = simplify(a0_integral)

print("\n" + "=" * 70)
print("FOURIER COEFFICIENTS")
print("=" * 70)

print(f"\na₀ = {a0}")
print(f"\nLaTeX: a₀ = {latex(a0)}")

# Compute an coefficients (cosine terms)
print("\n" + "-" * 70)
print("Cosine Coefficients (aₙ):")
print("-" * 70)

a_coeffs = {}
n_sym = symbols('n', integer=True, positive=True)

for n in range(1, N_TERMS + 1):
    print(f"Computing a_{n}...")
    an_integral = integrate(f * cos(n * pi * x / L), (x, -L, L)) / L
    an = simplify(an_integral)
    a_coeffs[n] = an
    
    if an != 0:
        print(f"  a_{n} = {an}")
        print(f"  LaTeX: a_{{{n}}} = {latex(an)}")
    else:
        print(f"  a_{n} = 0")

# Compute bn coefficients (sine terms)
print("\n" + "-" * 70)
print("Sine Coefficients (bₙ):")
print("-" * 70)

b_coeffs = {}

for n in range(1, N_TERMS + 1):
    print(f"Computing b_{n}...")
    bn_integral = integrate(f * sin(n * pi * x / L), (x, -L, L)) / L
    bn = simplify(bn_integral)
    b_coeffs[n] = bn
    
    if bn != 0:
        print(f"  b_{n} = {bn}")
        print(f"  LaTeX: b_{{{n}}} = {latex(bn)}")
    else:
        print(f"  b_{n} = 0")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY - FOURIER SERIES")
print("=" * 70)
print(f"\nf(x) ≈ a₀ + Σ[aₙcos(nπx/L) + bₙsin(nπx/L)]")
print(f"\nwhere L = {L}")
print(f"\na₀ = {a0}")

print("\nNon-zero cosine coefficients:")
for n, an in a_coeffs.items():
    if an != 0:
        print(f"  a_{n} = {an}")

print("\nNon-zero sine coefficients:")
for n, bn in b_coeffs.items():
    if bn != 0:
        print(f"  b_{n} = {bn}")

# ============================================================================
# NUMERICAL EVALUATION AND PLOTTING
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING INTERACTIVE PLOT")
print("=" * 70)

# Convert symbolic coefficients to numerical functions
a0_num = float(a0.evalf())
a_coeffs_num = {n: float(an.evalf()) for n, an in a_coeffs.items()}
b_coeffs_num = {n: float(bn.evalf()) for n, bn in b_coeffs.items()}

# Create numerical version of original function
f_lambda = sp.lambdify(x, f, 'numpy')

# Generate x values for plotting
x_vals = np.linspace(-float(L), float(L), 1000)
f_vals = f_lambda(x_vals)

# Function to compute Fourier series sum up to k terms
def fourier_sum(x_array, k):
    """Compute Fourier series sum up to k terms"""
    result = np.ones_like(x_array) * a0_num
    
    for n in range(1, min(k + 1, N_TERMS + 1)):
        if n in a_coeffs_num and a_coeffs_num[n] != 0:
            result += a_coeffs_num[n] * np.cos(n * np.pi * x_array / float(L))
        if n in b_coeffs_num and b_coeffs_num[n] != 0:
            result += b_coeffs_num[n] * np.sin(n * np.pi * x_array / float(L))
    
    return result

# Create the figure and axis
fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.25, left=0.08, right=0.95, top=0.92)

# Initial number of terms
k_initial = 5

# Plot original function with better styling
line_original, = ax.plot(x_vals, f_vals, 'b-', linewidth=2.5, label='Original f(x)', 
                         alpha=0.8, zorder=3)

# Plot Fourier series with better styling
fourier_vals = fourier_sum(x_vals, k_initial)
line_fourier, = ax.plot(x_vals, fourier_vals, 'r--', linewidth=2.5, 
                        label=f'Fourier Series (k={k_initial})', alpha=0.9, zorder=4)

# Enhanced grid
ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Labels and title with larger fonts
ax.set_xlabel('x', fontsize=14, fontweight='bold')
ax.set_ylabel('f(x)', fontsize=14, fontweight='bold')
ax.set_title(f'Fourier Series Approximation: f(x) = {f}', fontsize=16, fontweight='bold', pad=20)

# Enhanced legend
legend = ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=True)
legend.get_frame().set_linewidth(1.5)

ax.set_xlim(-float(L), float(L))

# Add some padding to y-limits
y_min, y_max = min(f_vals.min(), fourier_vals.min()), max(f_vals.max(), fourier_vals.max())
y_range = y_max - y_min
ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

# Improve tick labels
ax.tick_params(axis='both', which='major', labelsize=11)

# Create slider with better positioning
ax_slider = plt.axes([0.2, 0.08, 0.6, 0.04])
ax_slider.set_facecolor('#f0f0f0')
slider = Slider(
    ax=ax_slider,
    label='Number of Fourier terms (k)',
    valmin=0,
    valmax=N_TERMS,
    valinit=k_initial,
    valstep=1,
    color='#1f77b4',
    track_color='#d3d3d3'
)

# Update function for slider
def update(val):
    k = int(slider.val)
    fourier_vals = fourier_sum(x_vals, k)
    line_fourier.set_ydata(fourier_vals)
    line_fourier.set_label(f'Fourier Series (k={k})')
    
    # Update y-limits to accommodate new values
    y_min = min(f_vals.min(), fourier_vals.min())
    y_max = max(f_vals.max(), fourier_vals.max())
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    ax.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=True)
    fig.canvas.draw_idle()

slider.on_changed(update)

# Add main info box
info_text = f'''FOURIER SERIES PARAMETERS
Period: L ∈ [{-float(L):.3g}, {float(L):.3g}]
Constant term: a₀ = {a0_num:.4g}
Computed terms: N = {N_TERMS}'''

props_main = dict(boxstyle='round,pad=0.8', facecolor='#e8f4f8', alpha=0.9, edgecolor='#1f77b4', linewidth=2)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
        verticalalignment='top', bbox=props_main, family='monospace')

# Add coefficient info box (showing first few non-zero coefficients)
coeff_text = "FIRST NON-ZERO COEFFICIENTS\n"
coeff_text += "─" * 25 + "\n"

non_zero_a = [(n, an) for n, an in a_coeffs_num.items() if abs(an) > 1e-10]
non_zero_b = [(n, bn) for n, bn in b_coeffs_num.items() if abs(bn) > 1e-10]

if non_zero_a:
    coeff_text += "Cosine (aₙ):\n"
    for n, an in non_zero_a[:3]:  # Show first 3
        coeff_text += f"  a₍{n}₎ = {an:+.4g}\n"
    if len(non_zero_a) > 3:
        coeff_text += f"  ... ({len(non_zero_a)} total)\n"
    coeff_text += "\n"

if non_zero_b:
    coeff_text += "Sine (bₙ):\n"
    for n, bn in non_zero_b[:3]:  # Show first 3
        coeff_text += f"  b₍{n}₎ = {bn:+.4g}\n"
    if len(non_zero_b) > 3:
        coeff_text += f"  ... ({len(non_zero_b)} total)\n"

props_coeff = dict(boxstyle='round,pad=0.8', facecolor='#fff4e6', alpha=0.9, edgecolor='#ff7f0e', linewidth=2)
ax.text(0.98, 0.98, coeff_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
        verticalalignment='top', horizontalalignment='right', bbox=props_coeff, family='monospace')

plt.show()

print("\nPlot displayed successfully!")
print("Use the slider to adjust the number of Fourier series terms.")
print("\nLegend:")
print("  • Blue solid line: Original function f(x)")
print("  • Red dashed line: Fourier series approximation")
print("  • Move slider to see convergence with more terms")

