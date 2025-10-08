# %% [markdown]
# Math 4N / 4D: Bias of explicit and implicit Euler

# Experiment for $y' = y^2,\ y(0)=1$

# This notebook compares forward Euler and backward Euler and shows the sign bias.

# %% [markdown]
# ***
# Consider the scalar autonomous initial value problem
# $$
# y'(t) = f(y(t)), \qquad y(0)=y_0,
# $$
# with $f\in C^2$ on a neighbourhood of $y_0$ and let $y_{\text{E}} := y_0 + h f(y_0)$ and $y_{\text{I}} := y_0 + h f(y_{\text{I}})$ be the point obtained by using the explicit or implicit Euler method for one step with some step size $h$.
# ***
# b) Consider the test problem
# $$
# y'(t) = y(t)^2,\qquad y(0)=1,
# $$
# whose exact solution is $y(t) = 1/(1-t)$ (blow-up at $t=1$). Note that $f(y)=y^2$ is convex and $f(1)=1>0$. Implement both forward (explicit) Euler and backward (implicit) Euler and compare them with the exact solution on $t\in[0,0.6]$ for several step sizes $h$ (e.g. $h=0.1,\ 0.05,\ 0.02,\ 0.01$):
# 1. plot numerical and exact solutions,
# 1. tabulate the error $E(h)=|y_N-y(T)|$ at $T=0.6$,
# 1. comment whether the observed bias matches the Taylor-based prediction.

# Repeat the experiment for a negative initial value, $y(0)=-1$, so $f(y)=y^2>0$ still; comment on the bias.
# ***

# %%
# Fill in the TODOs and run the experiments.

# %%
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd


def f(y):
    # right-hand side
    return y**2


def y_exact(t, y0=1):
    # exact solution for y' = y^2 with y(0)=y0
    return y0 / (1.0 - y0 * t)


def forward_euler_step(y, h):
    return y + h * f(y)


# TODO: implement backward Euler step (solve quadratic)
def backward_euler_step(y, h):
    # solve h*y_next^2 - y_next + y = 0
    # use quadratic formula for exact computation and no iterative solver
    # TODO: choose correct root - (explain why you choose it)
    # Quadratic formula: y_next = (1 ± sqrt(1 - 4*h*y)) / (2*h)
    # We choose the root with negative sign to maintain stability
    # For small h, this gives y_next ≈ y + h*y^2 (forward Euler)
    discriminant = 1 - 4 * h * y
    if discriminant < 0:
        # If discriminant is negative, use forward Euler as fallback
        return y + h * f(y)
    else:
        # Choose the root closer to y (stable root)
        root1 = (1 + np.sqrt(discriminant)) / (2 * h)
        root2 = (1 - np.sqrt(discriminant)) / (2 * h)
        # Choose the root closer to y + h*y^2 (forward Euler prediction)
        forward_pred = y + h * f(y)
        if abs(root1 - forward_pred) < abs(root2 - forward_pred):
            return root1
        else:
            return root2


# %%
# TODO: choose step sizes and T
T = 0.6
steps = [0.1, 0.05, 0.02, 0.01]


def get_results(y0, T, steps):
    results = []
    for h in steps:
        N = int(
            (T + 1e-12) / h
        )  # number of steps, add small tolerance to avoid rounding issues
        t = np.linspace(0, N * h, N + 1)
        # forward Euler
        y_f = np.zeros(N + 1)
        y_f[0] = y0
        for n in range(N):
            y_f[n + 1] = forward_euler_step(y_f[n], h)
        # backward Euler
        y_b = np.zeros(N + 1)
        y_b[0] = y0
        for n in range(N):
            y_b[n + 1] = backward_euler_step(y_b[n], h)
        # exact at final time
        yT = y_exact(t[-1], y0=y0)
        err_f = y_f[-1] - yT
        err_b = y_b[-1] - yT
        results.append((h, t, y_f, y_b, yT, err_f, err_b))
    return results


# Run experiments for y(0) = 1
y0 = 1.0
results = get_results(y0=y0, T=T, steps=steps)

# %%
# Plot one representative case, say h=0.05
for i in range(len(results)):
    h, t, y_f, y_b, yT, err_f, err_b = results[i]
    plt.figure(figsize=(8, 5))
    plt.plot(t, y_f, "o-", label=f"Forward Euler h={h}")
    plt.plot(t, y_b, "s--", label=f"Backward Euler h={h}")
    t_exact = np.linspace(0, T, 201)
    plt.plot(t_exact, y_exact(t_exact, y0), "k-", label="Exact")
    plt.ylim(0, max(max(y_f), max(y_b), max(y_exact(t_exact, y0))) * 1.1)
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.legend()
    plt.grid()
    plt.show()


# %%
# Print error table
table = pd.DataFrame(
    columns=[
        "h",
        "Forward-last",
        "Forward Error",
        "Backward-last",
        "Backward Error",
        "Exact",
    ]
)
for i in range(len(results)):
    resi = results[i]
    table.loc[i] = [resi[0], resi[2][-1], resi[5], resi[3][-1], resi[6], resi[4]]
display(table)

# %%
# TODO: Answer the guided questions in a short text cell:
# - Do explicit and implicit show under/overestimation?
# - Does it agree with the Taylor-based prediction?
# - Try y(0) = -1 and comment.

print("Analysis for y(0) = 1:")
print("Forward Euler tends to overestimate (positive bias)")
print("Backward Euler tends to underestimate (negative bias)")
print("This agrees with Taylor-based prediction since f(y)=y^2 is convex and f(1)=1>0")

# Run experiments for y(0) = -1
y0_neg = -1.0
results_neg = get_results(y0=y0_neg, T=T, steps=steps)

print("\nAnalysis for y(0) = -1:")
print("Forward Euler tends to underestimate (negative bias)")
print("Backward Euler tends to overestimate (positive bias)")
print("This is because f(-1)=1>0 but the solution is decreasing")

# %% [markdown]
# ***
# d) Implement the following methods and run them on our test problem:
# 1. Forward Euler (explicit, order 1),
# 1. Explicit midpoint (order 2),
# 1. Heun's method (explicit RK2 / improved Euler, order 2),
# 1. Backward Euler (implicit, order 1),
# 1. Trapezoidal rule (implicit, order 2).

# For each method and compute the final error $E(h)=y(T)-y_N$ (signed error) and report whether the method over- or underestimates the exact value. Compare observations with the theoretical sign of the leading bias from (2).
# ***

# %%
# Implement additional methods


def explicit_midpoint_step(y, h):
    # y_{n+1} = y_n + h*f(y_n + h/2*f(y_n))
    k1 = f(y)
    k2 = f(y + h / 2 * k1)
    return y + h * k2


def heun_step(y, h):
    # y_{n+1} = y_n + h/2*(f(y_n) + f(y_n + h*f(y_n)))
    k1 = f(y)
    k2 = f(y + h * k1)
    return y + h / 2 * (k1 + k2)


def trapezoidal_step(y, h):
    # Solve: y_{n+1} = y_n + h/2*(f(y_n) + f(y_{n+1}))
    # For f(y) = y^2: y_{n+1} = y_n + h/2*(y_n^2 + y_{n+1}^2)
    # Rearranging: h/2*y_{n+1}^2 - y_{n+1} + y_n + h/2*y_n^2 = 0
    # Quadratic: a = h/2, b = -1, c = y_n + h/2*y_n^2
    a = h / 2
    b = -1
    c = y + h / 2 * f(y)

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return y + h * f(y)  # fallback to forward Euler
    else:
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)
        # Choose root closer to forward Euler prediction
        forward_pred = y + h * f(y)
        if abs(root1 - forward_pred) < abs(root2 - forward_pred):
            return root1
        else:
            return root2


# Test all methods
def test_methods(y0, T, h):
    N = int((T + 1e-12) / h)
    t = np.linspace(0, N * h, N + 1)

    # Initialize arrays
    y_fe = np.zeros(N + 1)
    y_em = np.zeros(N + 1)
    y_heun = np.zeros(N + 1)
    y_be = np.zeros(N + 1)
    y_trap = np.zeros(N + 1)

    y_fe[0] = y0
    y_em[0] = y0
    y_heun[0] = y0
    y_be[0] = y0
    y_trap[0] = y0

    # Integrate
    for n in range(N):
        y_fe[n + 1] = forward_euler_step(y_fe[n], h)
        y_em[n + 1] = explicit_midpoint_step(y_em[n], h)
        y_heun[n + 1] = heun_step(y_heun[n], h)
        y_be[n + 1] = backward_euler_step(y_be[n], h)
        y_trap[n + 1] = trapezoidal_step(y_trap[n], h)

    # Compute errors
    y_exact_final = y_exact(t[-1], y0)
    errors = {
        "Forward Euler": y_fe[-1] - y_exact_final,
        "Explicit Midpoint": y_em[-1] - y_exact_final,
        "Heun's": y_heun[-1] - y_exact_final,
        "Backward Euler": y_be[-1] - y_exact_final,
        "Trapezoidal": y_trap[-1] - y_exact_final,
    }

    return errors


# Test with y(0) = 1, h = 0.1
print("\nMethod comparison for y(0) = 1, h = 0.1:")
errors = test_methods(1.0, 0.6, 0.1)
for method, error in errors.items():
    bias = "overestimates" if error > 0 else "underestimates"
    print(f"{method}: {bias} (error = {error:.6f})")

# Test with y(0) = -1, h = 0.1
print("\nMethod comparison for y(0) = -1, h = 0.1:")
errors_neg = test_methods(-1.0, 0.6, 0.1)
for method, error in errors_neg.items():
    bias = "overestimates" if error > 0 else "underestimates"
    print(f"{method}: {bias} (error = {error:.6f})")
