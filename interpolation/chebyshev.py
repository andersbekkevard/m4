# ------------------------------------------------------------
# Intuition builder for Chebyshev polynomials via cos/arccos:
# - Plot cos(nθ) vs θ
# - Show mapping x = cos θ
# - Plot T_n(x) = cos(n arccos x) vs x
# - Mark roots/extrema; pretty-print T_n(x) in monomial form
#
# Usage: just run:  python cheby_mapping_demo.py
# Tune the "CONFIG" section to try different n, samples, etc.
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev, Polynomial
import textwrap

# ---------------------- CONFIG ----------------------
n = 2  # degree of T_n
num_samples = 2000  # resolution for smooth plots
save_fig = False  # set True to save PNG
fig_path = f"cheby_T{n}_mapping.png"
# ----------------------------------------------------

# Safety checks
if n < 0 or int(n) != n:
    raise ValueError("n must be a nonnegative integer")

# 1) Define theta and x grids
theta = np.linspace(0.0, np.pi, num_samples)  # angle space
x = np.cos(theta)  # projection to [-1, 1]

# 2) The 'pure wave' and the pulled-back polynomial
cos_n_theta = np.cos(n * theta)  # cos(nθ)
# Using the defining relation: T_n(x) = cos(n arccos x)
# For plotting, just evaluate via the definition
Tn_from_def = np.cos(n * np.arccos(x))

# 3) Build Chebyshev polynomial T_n as an actual polynomial in x
# Chebyshev basis -> Polynomial (monomial) basis
Tn_cheb = Chebyshev.basis(n)  # T_n in Chebyshev basis
Tn_poly = Tn_cheb.convert(kind=Polynomial)  # convert to monomial basis
coeffs = Tn_poly.coef  # [c0, c1, c2, ...] low -> high degree

# 4) Roots and extrema of T_n
# Roots: cos((2k-1)*pi/(2n)), k=1..n (if n>=1)
roots = None
if n >= 1:
    k = np.arange(1, n + 1)
    roots = np.cos((2 * k - 1) * np.pi / (2 * n))

# Extrema locations: cos(k*pi/n), k=0..n
extrema_x = np.cos(np.arange(0, n + 1) * np.pi / n) if n >= 1 else np.array([1.0])
extrema_vals = np.cos(n * np.arccos(extrema_x))  # will be ±1


# 5) Pretty-print T_n(x) in monomial form
def pretty_polynomial(coefs, var="x", digits=10):
    """
    Turn [c0, c1, ..., cN] into a human-friendly string:
      cN x^N + ... + c2 x^2 + c1 x + c0
    with nice sign handling and rounded coefficients.
    """
    terms = []
    N = len(coefs) - 1
    for power in range(N, -1, -1):
        c = round(float(coefs[power]), digits)
        if abs(c) < 1e-14:
            continue  # skip tiny coefficients
        # Sign and magnitude
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        # Term text
        if power == 0:
            term = f"{mag:g}"
        elif power == 1:
            if np.isclose(mag, 1.0):
                term = var
            else:
                term = f"{mag:g}{var}"
        else:
            if np.isclose(mag, 1.0):
                term = f"{var}^{power}"
            else:
                term = f"{mag:g}{var}^{power}"

        terms.append((sign, term))

    if not terms:
        return "0"

    # Build string with correct leading sign
    first_sign, first_term = terms[0]
    s = ("-" if first_sign == "-" else "") + first_term
    for sign, term in terms[1:]:
        s += f" {sign} {term}"
    return s


# Prepare the title text
title_text = f"Chebyshev Intuition via cos ↔ arccos  (T_{n}(x))"

# 6) Make the figure
plt.figure(figsize=(11, 10))

# (A) cos(nθ) vs θ
ax1 = plt.subplot(3, 1, 1)
ax1.plot(theta, cos_n_theta, linewidth=2)
ax1.set_title(
    r"(A) Pure wave in angle-space: $\,\cos(n\theta)\,$ vs. $\,\theta\in[0,\pi]$"
)
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$\cos(n\theta)$")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-1.1, 1.1)

# (B) mapping x = cos θ
ax2 = plt.subplot(3, 1, 2)
ax2.plot(theta, x, linewidth=2)
ax2.set_title(r"(B) Mapping: $\,x=\cos\theta\,$ (projects $[0,\pi]$ onto $[-1,1]$)")
ax2.set_xlabel(r"$\theta$")
ax2.set_ylabel(r"$x=\cos\theta$")
ax2.grid(True, alpha=0.3)

# (C) T_n(x) vs x
ax3 = plt.subplot(3, 1, 3)
ax3.plot(x, Tn_from_def, linewidth=2, label=rf"$T_{n}(x)=\cos(n\arccos x)$")
ax3.set_title(r"(C) Pulled-back polynomial: $\,T_n(x)\,$ vs. $\,x\in[-1,1]$")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$T_n(x)$")
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1.0, 1.0)
ax3.set_ylim(-1.1, 1.1)

# Overlay roots and extrema on panel (C)
if n >= 1 and roots is not None:
    ax3.scatter(
        roots, np.zeros_like(roots), marker="o", s=45, label="roots of $T_n$", zorder=5
    )
ax3.scatter(extrema_x, extrema_vals, marker="s", s=45, label="extrema (±1)", zorder=5)

# Aesthetics
ax3.legend(loc="upper center", ncol=3, frameon=True)
plt.suptitle(title_text, y=0.98, fontsize=14)

plt.tight_layout(rect=(0, 0, 1, 0.96))

if save_fig:
    plt.savefig(fig_path, dpi=200)
    print(f"Saved figure to: {fig_path}")

# 7) Print polynomial information
print("\n" + "=" * 72)
print(f"Chebyshev polynomial T_{n}(x) as a true polynomial in x (monomial basis):")
print("-" * 72)
poly_str = pretty_polynomial(coeffs, var="x", digits=12)
print(textwrap.fill(poly_str, width=78))
print("-" * 72)
print("Coefficients (low→high degree):")
for i, c in enumerate(coeffs):
    print(f"  x^{i}: {c:.12g}")
print("=" * 72 + "\n")

# 8) Sanity check: Evaluate Tn_poly(x) and compare to definition
x_test = np.linspace(-1, 1, 9)
val_poly = Tn_poly(x_test)
val_def = np.cos(n * np.arccos(x_test))
max_abs_diff = np.max(np.abs(val_poly - val_def))
print(
    f"Sanity check on grid (monomial poly vs definition): max |Δ| = {max_abs_diff:.3e}"
)

plt.show()
