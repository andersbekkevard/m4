# %% [markdown]
# Numerical solution of ODEs
# In this problem we will implement Euler's method, second order Taylor's method, and Heun's method, and use them to approximate the solution to the ODE,
# \begin{align*}
# 	y' = (1-2t) y, \quad y(0) = 1.
# \end{align*}
# The exact solution to this equation is $y(t) = \mathrm{e}^{t-t^2}$.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# define the derivatives of y


def d_y(t, y):
    # first derivative of y
    return (1 - 2 * t) * y


def dd_y(t, y):
    # second derivative (for second order Taylor method)
    # d/dt[(1-2t)y] = -2y + (1-2t)*dy/dt = -2y + (1-2t)^2*y
    return -2 * y + (1 - 2 * t) ** 2 * y


# define y


def y(t):
    # y(t) (for error computations)
    return np.exp(t - t**2)


# %% [markdown]
# >***
# >a) Implement Euler's method, and compute an approximation of $y(1)$, using a step size equal to $0.5$.
# >***


# %%
# Define Euler's method
def Euler(y0, T, f, h):
    """
        Euler(y0, T, f, h)

    Use Euler's method to approximate the solution of the ODE (scalar or system) y'(t) = f(t,y)

    Input:
        y0 - initial value y(0)
        f  - definition of the right-hand-side function
        T  - simulation time (starting at t=0, ending at t=T)
        h  - time-step size (fixed)
    Output:
        y  - array containing all discrete solution values y0,y1,y2,...
        t  - array containing all discrete time instants 0,h,2h,...
    """
    ys = [y0]  # array where all y_n will be stored
    ts = [0]  # array where all t_n will be stored
    while ts[-1] < T:
        t, y = ts[-1], ys[-1]
        ys.append(y + h * f(t, y))
        ts.append(t + h)
    return (np.array(ts), np.array(ys))


# %% [markdown]
# >***
# >b) Do the same using Heun's method and the second order Taylor method. The second order Taylor method is given by
# >$$y_{n+1} = y_n + hf(t_n,y_n) + \frac{h^2}{2}f'(t_n,y_n),$$
# >where $f(t,y) = y'(t)$ and thus $f'(t,y) := \frac{\partial f}{\partial t}(t,y) = y''(t)$.
# >***


# %%
# Define Heun's method
def Heun(y0, T, f, h):
    """
        Heun(y0, T, f, h)

    Use Heun's method to approximate the solution of the ODE (scalar or system) y'(t) = f(t,y)

    Input:
        y0 - initial value y(0)
        f  - definition of the right-hand-side function
        T  - simulation time (starting at t=0, ending at t=T)
        h  - time-step size (fixed)
    Output:
        y  - array containing all discrete solution values y0,y1,y2,...
        t  - array containing all discrete time instants 0,h,2h,...
    """
    ys = [y0]  # array where all y_n will be stored
    ts = [0]  # array where all t_n will be stored
    while ts[-1] < T:
        t, y = ts[-1], ys[-1]
        # Predictor step
        y_pred = y + h * f(t, y)
        # Corrector step
        ys.append(y + (h / 2) * (f(t, y) + f(t + h, y_pred)))
        ts.append(t + h)
    return (np.array(ts), np.array(ys))


# %%
# Define the second-order Taylor method
def scalarTaylor2(y0, T, f, DfDt, h):
    """
        scalarTaylor2(y0, T, f, h)

    Use the second-order Taylor method to approximate the solution of the scalar ODE y'(t) = f(t,y)

    Input:
        y0   - initial value y(0)
        f    - definition of the right-hand-side function
        DfDt - definition of the total time derivative of f
        T    - simulation time (starting at t=0, ending at t=T)
        h    - time-step size (fixed)
    Output:
        y  - vector containing all discrete solution values y0,y1,y2,...
        t  - vector containing all discrete time instants 0,h,2h,...
    """
    ys = [y0]  # array where all y_n will be stored
    ts = [0]  # array where all t_n will be stored
    while ts[-1] < T:
        t, y = ts[-1], ys[-1]
        ys.append(y + h * f(t, y) + 0.5 * h * h * DfDt(t, y))
        ts.append(t + h)
    return (np.array(ts), np.array(ys))


# %%
# compute the approximations using the different methods and h = 0.5

# exact value of y(1)
print(f"Exact y(1): {y(1.0)}")

# y(1) for Euler's method
euler_result = Euler(1.0, 1.0, d_y, 0.5)
print(f"Euler y(1): {euler_result[1][-1]}")

# y(1) for Heun's method
heun_result = Heun(1.0, 1.0, d_y, 0.5)
print(f"Heun y(1): {heun_result[1][-1]}")

# y(1) for second order Taylor's method
taylor_result = scalarTaylor2(1.0, 1.0, d_y, dd_y, 0.5)
print(f"Taylor2 y(1): {taylor_result[1][-1]}")

# %% [markdown]
# >***
# >c) We now want to approximate the convergence orders of these methods numerically. Recall that we defined the global error,
# \begin{align*}
# 	\epsilon_g := \max_i \lvert y(t_i) - y_i \rvert.
# \end{align*}
# >If we assume that $\epsilon_g(h) \approx M h^p$, for some $M > 0$, we have,
# \begin{align*}
# 	\log \bigg(\frac{\epsilon_g(h_1)}{\epsilon_g(h_2)} \bigg) \approx p \log \bigg(\frac{h_1}{h_2} \bigg).
# \end{align*}
# >Compute the global error of the methods from a)-c) using $h_1 = 10^{-2}$ and $h_2 = 10^{-3}$, where $t_i = i h, i = 1, \dots, \frac{1}{h}$. Use this to approximate the convergence order, $p$, for each of the three methods.
# >***

# %%
# approximate convergence order for each method

h1 = 1e-2

h2 = 1e-3

# Euler's method
ts, ys = Euler(1.0, 1.0, d_y, h1)
e1 = np.linalg.norm(ys - y(ts), np.inf)
ts2, ys2 = Euler(1.0, 1.0, d_y, h2)
e2 = np.linalg.norm(ys2 - y(ts2), np.inf)
p_Euler = np.log(e1 / e2) / np.log(h1 / h2)
print(f"Euler's method convergence order: {p_Euler}")

# Heun's method
ts, ys = Heun(1.0, 1.0, d_y, h1)
e1 = np.linalg.norm(ys - y(ts), np.inf)
ts2, ys2 = Heun(1.0, 1.0, d_y, h2)
e2 = np.linalg.norm(ys2 - y(ts2), np.inf)
p_Heun = np.log(e1 / e2) / np.log(h1 / h2)
print(f"Heun's method convergence order: {p_Heun}")

# second order Taylor's method
ts, ys = scalarTaylor2(1.0, 1.0, d_y, dd_y, h1)
e1 = np.linalg.norm(ys - y(ts), np.inf)
ts2, ys2 = scalarTaylor2(1.0, 1.0, d_y, dd_y, h2)
e2 = np.linalg.norm(ys2 - y(ts2), np.inf)
p_Taylor2 = np.log(e1 / e2) / np.log(h1 / h2)
print(f"Second-order Taylor method convergence order: {p_Taylor2}")

# %% [markdown]
# >***
# >d) We can also approximate the convergence order by plotting $\log(\epsilon_g(h)) = \log(M) + p \log(h)$ versus $\log(h)$, and inspecting the slope of the function.\\
# >Plot $\log(\epsilon_g(h))$ versus $\log(h)$ for $h = 10^{-2}, 10^{-3}, 10^{-4}$ for each of the three methods.
# >***

# %%
# plot log(error) versus log(h)

# values of h
hs = [1e-2, 1e-3, 1e-4]

errors_Euler = []
errors_Heun = []
errors_Taylor2 = []
for h in hs:
    # Euler's method error
    ts, ys = Euler(1.0, 1.0, d_y, h)
    error = np.linalg.norm(ys - y(ts), np.inf)
    errors_Euler.append(error)

    # Heun's method error
    ts, ys = Heun(1.0, 1.0, d_y, h)
    error = np.linalg.norm(ys - y(ts), np.inf)
    errors_Heun.append(error)

    # second order Taylor method error
    ts, ys = scalarTaylor2(1.0, 1.0, d_y, dd_y, h)
    error = np.linalg.norm(ys - y(ts), np.inf)
    errors_Taylor2.append(error)

# plot of Euler's method errors
plt.plot(np.log(hs), np.log(errors_Euler), label="Euler's method")

# plot of Heun's method errors
plt.plot(np.log(hs), np.log(errors_Heun), label="Heun's method")

# plot of second order Taylor error
plt.plot(np.log(hs), np.log(errors_Taylor2), label="Taylor's method")

plt.xlabel("log(h)")
plt.ylabel("log(error)")
plt.legend()
plt.grid(True)
