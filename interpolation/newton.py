from collections.abc import Callable
from math import prod, sin
import numpy as np
import matplotlib.pyplot as plt
import inspect


def newton_coefficient(f, x: list):
    return _newton_coefficient_sorted(f, sorted(x, reverse=False))


def _newton_coefficient_sorted(f, x: list):
    if len(x) == 1:
        return f(x[0])
    else:
        return (
            _newton_coefficient_sorted(f, x[:-1]) - _newton_coefficient_sorted(f, x[1:])
        ) / (x[0] - x[-1])


def newton_coefficients(f, x: list):
    if len(x) == 1:
        return [f(x[0])]
    else:
        # Build coefficients in ascending order: [f[x₀], f[x₀,x₁], f[x₀,x₁,x₂], ...]
        return [newton_coefficient(f, x[: i + 1]) for i in range(len(x))]


def newton_polynomial(f, x: list):
    coeffs = newton_coefficients(f, x)

    def poly(x_val):
        result = 0
        for i, coeff in enumerate(coeffs):
            term = coeff
            for j in range(i):
                term *= x_val - x[j]
            result += term
        return result

    return poly


def f(x):
    return sin(x)


# Test cases
print("=" * 60)
print("NEWTON INTERPOLATION TESTING SUITE")
print("=" * 60)

# Test 2nd order (2 points)
x = [1, 2]
coeff_2 = newton_coefficient(f, x)
actual_2 = f(2)
print(f"\n🔍 2nd Order Interpolation (2 points):")
print(f"   Points: x = {x}")
print(f"   Newton coefficient: {coeff_2:.6f}")
print(f"   f({x[1]}) = {actual_2:.6f}")

# Test 3rd order (3 points)
x3 = [1, 2, 3]
coeff_3 = newton_coefficient(f, x3)
actual_3 = f(3)
print(f"\n🔍 3rd Order Interpolation (3 points):")
print(f"   Points: x = {x3}")
print(f"   Newton coefficient: {coeff_3:.6f}")
print(f"   f({x3[2]}) = {actual_3:.6f}")

# Test 4th order (4 points)
x4 = [1, 2, 3, 4]
coeff_4 = newton_coefficient(f, x4)
actual_4 = f(4)
print(f"\n🔍 4th Order Interpolation (4 points):")
print(f"   Points: x = {x4}")
print(f"   Newton coefficient: {coeff_4:.6f}")
print(f"   f({x4[3]}) = {actual_4:.6f}")

# Test the full polynomial
print(f"\n" + "=" * 60)
print("FULL POLYNOMIAL INTERPOLATION TEST")
print("=" * 60)

interpolation_points = [1, 2, 3]
print(f"Interpolation points: x = {interpolation_points}")
print(f"Function: f(x) = sin(x)")

# Get Newton coefficients
newton_coeffs = newton_coefficients(f, interpolation_points)
print(f"\n📊 Newton coefficients: {[f'{c:.6f}' for c in newton_coeffs]}")

# Debug: Show what each coefficient represents
print(f"\n🔍 Coefficient Analysis:")
for i, coeff in enumerate(newton_coeffs):
    if i == 0:
        print(f"   c₀ = f[x₀] = f({interpolation_points[0]}) = {coeff:.6f}")
    else:
        points_used = interpolation_points[: i + 1]
        print(f"   c{i} = f[x₀,x₁,...,x{i}] = f{points_used} = {coeff:.6f}")

# Create polynomial
poly = newton_polynomial(f, interpolation_points)

# Test polynomial at various points
test_points = [1.5, 2.5, 3.5]
print(f"\n🧪 Polynomial Evaluation Tests:")
print(f"{'x':>6} {'Polynomial':>15} {'Actual f(x)':>15} {'Difference':>15}")
print("-" * 60)

for x_test in test_points:
    poly_result = poly(x_test)
    actual_result = f(x_test)
    difference = abs(poly_result - actual_result)
    print(
        f"{x_test:>6.1f} {poly_result:>15.6f} {actual_result:>15.6f} {difference:>15.6f}"
    )

# Test at interpolation points (should be exact)
print(f"\n🎯 Verification at Interpolation Points:")
print(f"{'x':>6} {'Polynomial':>15} {'Actual f(x)':>15} {'Exact Match?':>15}")
print("-" * 60)

for x_point in interpolation_points:
    poly_result = poly(x_point)
    actual_result = f(x_point)
    is_exact = abs(poly_result - actual_result) < 1e-10
    match_status = "✅ YES" if is_exact else "❌ NO"
    print(
        f"{x_point:>6.0f} {poly_result:>15.6f} {actual_result:>15.6f} {match_status:>15}"
    )

print(f"\n" + "=" * 60)

# Plotting test
print("\n📈 Creating visualization plot...")

# Create x values for plotting
x_plot = np.linspace(0, 4, 100)

# Calculate y values for original function and Newton polynomial
y_original = [f(x_val) for x_val in x_plot]
y_newton = [poly(x_val) for x_val in x_plot]

# Get function name dynamically
func_name = inspect.getsource(f).strip().split("return ")[-1].strip()

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(
    x_plot, y_original, "b-", label=f"Original function f(x) = {func_name}", linewidth=2
)
plt.plot(x_plot, y_newton, "r--", label="Newton interpolation polynomial", linewidth=2)
plt.scatter(
    interpolation_points,
    [f(x) for x in interpolation_points],
    color="green",
    s=100,
    zorder=5,
    label="Interpolation points",
)

# Customize the plot
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Newton Interpolation: f(x) = {func_name}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()

print("✅ Plot completed!")
print("=" * 60)
